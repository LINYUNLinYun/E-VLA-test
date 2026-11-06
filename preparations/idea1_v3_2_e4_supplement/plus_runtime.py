"""Extract pinned official apt dependencies privately; never install system packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import posixpath
import subprocess
import tarfile
import traceback
from .native_envs import write_json,sha

PACKAGES={
    'libfftw3-double3':'3.3.8-2ubuntu8','libde265-0':'1.0.8-1ubuntu0.3','libheif1':'1.12.0-2build1',
    'liblqr-1-0':'0.4.2-2.1','libltdl7':'2.4.6-15build2','libwebpdemux2':'1.2.2-2ubuntu0.22.04.2',
    'imagemagick-6-common':'8:6.9.11.60+dfsg-1.3ubuntu0.22.04.5',
    'libmagickcore-6.q16-6':'8:6.9.11.60+dfsg-1.3ubuntu0.22.04.5',
    'libmagickwand-6.q16-6':'8:6.9.11.60+dfsg-1.3ubuntu0.22.04.5'}


def audit_archive(path):
    process=subprocess.Popen(['dpkg-deb','--fsys-tarfile',str(path)],stdout=subprocess.PIPE)
    count=0
    try:
        with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
            for member in archive:
                name=PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts: raise ValueError('Unsafe deb member path')
                if member.issym() or member.islnk():
                    target=PurePosixPath(member.linkname)
                    resolved=posixpath.normpath(str(name.parent/target)) if member.issym() else posixpath.normpath(str(target))
                    if target.is_absolute() or resolved.startswith('../') or resolved=='..':
                        raise ValueError('Deb link escapes private runtime: '+member.name+' -> '+member.linkname)
                if member.ischr() or member.isblk() or member.isfifo(): raise ValueError('Special filesystem node forbidden')
                count+=1
        if process.wait()!=0: raise RuntimeError('dpkg-deb archive stream failed')
    finally:
        if process.poll() is None: process.terminate();process.wait()
    return count


def prepare(downloads,receipt_path):
    downloads=Path(downloads).resolve();repository=Path('/policies/LIBERO-plus')
    if downloads.parent!=repository or not downloads.name.startswith('.runtime-debs.'):
        raise ValueError('Use a newly allocated download directory inside only LIBERO-plus')
    destination=repository/'.runtime'
    if destination.exists(): raise FileExistsError('Never overwrite an existing private runtime')
    staging=repository/('.runtime.staging-%d'%os.getpid())
    receipt=dict(status='auditing',download_directory=str(downloads),destination=str(destination),
        process_pid=os.getpid(),system_packages_installed=False,system_upgrades=False,
        download_command=['apt-get','download']+[name+'='+version for name,version in PACKAGES.items()],archives=[],steps=[])
    write_json(receipt_path,receipt)
    try:
        found={}
        for path in sorted(downloads.glob('*.deb')):
            metadata=subprocess.check_output(['dpkg-deb','-f',str(path),'Package','Version'],universal_newlines=True)
            fields=dict(line.split(': ',1) for line in metadata.splitlines() if ': ' in line)
            name,version=fields['Package'],fields['Version']
            if PACKAGES.get(name)!=version or name in found: raise ValueError('Unexpected/duplicate pinned apt package')
            apt=subprocess.check_output(['apt-cache','show',name+'='+version],universal_newlines=True)
            expected=next(line.split(': ',1)[1] for line in apt.splitlines() if line.startswith('SHA256: '))
            actual=sha(path)
            if expected!=actual: raise ValueError('Archive differs from configured official apt metadata')
            count=audit_archive(path)
            row=dict(package=name,version=version,path=str(path),bytes=path.stat().st_size,sha256=actual,
                configured_apt_sha256=expected,safe_members=count)
            found[name]=path;receipt['archives'].append(row);write_json(receipt_path,receipt)
        if set(found)!=set(PACKAGES): raise ValueError('Pinned private runtime download set incomplete')
        staging.mkdir(exist_ok=False);receipt.update(status='extracting',staging=str(staging));write_json(receipt_path,receipt)
        for name in PACKAGES:
            command=['dpkg-deb','--extract',str(found[name]),str(staging)]
            result=subprocess.run(command,capture_output=True,text=True)
            receipt['steps'].append(dict(command=command,returncode=result.returncode,stderr=result.stderr))
            write_json(receipt_path,receipt)
            if result.returncode: raise RuntimeError('Private package extraction failed')
        # Wand's MAGICK_HOME search uses prefix/lib; Ubuntu ships multiarch libs.
        library=staging/'usr/lib'
        for path in sorted((library/'x86_64-linux-gnu').glob('libMagick*.so.6')):
            (library/path.name).symlink_to(Path('x86_64-linux-gnu')/path.name)
        staging.rename(destination)
        receipt.update(status='complete',runtime=str(destination),
            child_environment=dict(MAGICK_HOME=str(destination/'usr'),
                LD_LIBRARY_PATH=str(destination/'usr/lib/x86_64-linux-gnu'),
                MAGICK_CONFIGURE_PATH=str(destination/'etc/ImageMagick-6'),
                MAGICK_CODER_MODULE_PATH=str(destination/'usr/lib/x86_64-linux-gnu/ImageMagick-6.9.11/modules-Q16/coders')))
    except Exception:
        receipt.update(status='failed',error=traceback.format_exc());raise
    finally: write_json(receipt_path,receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--downloads',required=True);parser.add_argument('--receipt',required=True)
    parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch: print(json.dumps(dict(status=prepare(args.downloads,args.receipt)['status'])))
    else: print(json.dumps(dict(dry_run=True,**vars(args))))
