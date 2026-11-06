"""Read-only graphics-process guard; compute-apps alone misses EGL rendering."""
import subprocess
import xml.etree.ElementTree as ET


def parse_graphics_processes(xml, uuid):
    devices = [node for node in ET.fromstring(xml).findall("gpu")
               if node.findtext("uuid") == uuid]
    if len(devices) != 1:
        raise RuntimeError("GPU XML does not uniquely match the requested UUID")
    rows = []
    for node in devices[0].findall("processes/process_info"):
        kind = node.findtext("type", "")
        if "G" not in kind:
            continue
        used = node.findtext("used_memory", "N/A").split()[0]
        rows.append(dict(pid=int(node.findtext("pid")), type=kind,
                         memory_mib=float(used) if used != "N/A" else None))
    return rows


def graphics_processes(uuid):
    result = subprocess.run(["nvidia-smi", "-i", uuid, "-q", "-x"],
                            check=True, capture_output=True, text=True)
    return parse_graphics_processes(result.stdout, uuid)
