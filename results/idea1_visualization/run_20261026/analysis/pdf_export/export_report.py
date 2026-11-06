"""Export the frozen Markdown report and all its static figures to one PDF."""
from pathlib import Path
import hashlib
import html
import json
import re
import sys

sys.path.insert(0,'/tmp/idea1_pdf_export_deps')
from markdown_it import MarkdownIt
from pypdf import PdfReader
from pypdf.generic import ContentStream
from weasyprint import HTML

WORK=Path(__file__).resolve().parent
ROOT=WORK.parent
SOURCE=ROOT/'report_zh.md'
OUTPUT=ROOT/'report_zh.pdf'
source=SOURCE.read_text()
markdown=MarkdownIt('commonmark',{'html':True}).enable('table')
body_start=source.index('## 逐层读出曲线')
main=markdown.render(source[:body_start])
title=html.unescape(re.search(r'<h1>(.*?)</h1>',main).group(1))
main=re.sub(r'<h1>.*?</h1>\s*','',main,count=1)
toc=[]
def heading(match):
    tag,text=match.groups();identifier=f'text-section-{len(toc)}'
    if tag=='h2':toc.append((identifier,html.unescape(re.sub('<[^>]+>','',text))))
    return f'<{tag} id="{identifier}">{text}</{tag}>'
main=re.sub(r'<(h2)>(.*?)</\1>',heading,main)

chapters=[]
headings=list(re.finditer(r'^## (.+)$',source[body_start:],re.M))
tail=source[body_start:]
assets=[]
for index,match in enumerate(headings):
    name=match.group(1)
    section=tail[match.end():headings[index+1].start() if index+1<len(headings) else len(tail)]
    paths=re.findall(r'!?\[[^\]]*\]\((figures/[^)]+\.png)\)',section)
    assert paths,name
    identifier=f'figure-section-{index}'
    toc.append((identifier,name))
    paragraphs=section[:section.find('![')].strip() if '![' in section else ''
    for i,relative in enumerate(paths):
        image=ROOT/relative;assert image.is_file(),image
        assets.append(relative)
        kind='wide' if image.name.startswith(('probe_','residual_')) else 'tall'
        feature=image.stem
        feature=feature.replace('pca_','PCA · ').replace('probe_','Probe · ').replace('conditional_','同 task 内 phase · ').replace('residual_','残差 · ')
        feature=feature.replace('vision_base_0_rgb','主相机 Vision').replace('vision_left_wrist_0_rgb','腕部相机 Vision').replace('vlm_vision','VLM 视觉池').replace('vlm_language','VLM 语言池')
        feature=feature.replace('_hidden',' · hidden').replace('_attn_update',' · attention 更新').replace('_ffn_update',' · FFN 更新').replace('_post_attention',' · attention 后状态').replace('_layer',' · H')
        feature=feature.replace('_block_relative_norm',' · block 相对范数').replace('_attn_relative_norm',' · attention 相对范数').replace('_ffn_relative_norm',' · FFN 相对范数').replace('_block_cosine',' · block 余弦')
        chapter=f'<h2 id="{identifier}">{html.escape(name)}</h2>' if i==0 else ''
        explanation=f'<div class="figure-note">{markdown.render(paragraphs)}</div>' if i==0 and paragraphs else ''
        chapters.append(f'<section class="plot-page {kind}">{chapter}<h3>图 {len(assets):02d} · {html.escape(feature)}</h3>{explanation}<div class="filename">{html.escape(image.name)}</div><img src="{image.as_uri()}" alt="{html.escape(feature)}"></section>')
assert len(assets)==68 and len(set(assets))==68
nav=''.join(f'<li><a href="#{identifier}">{html.escape(name)}</a></li>' for identifier,name in toc)
css='''
@font-face{font-family:CJK;src:url(file:///usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf)}
@page{size:A4;margin:17mm 16mm 17mm;
 @top-left{content:"π0.5 · LIBERO 表征探索";font:8pt sans-serif;color:#64748b}
 @bottom-center{content:"第 " counter(page) " 页";font:8pt sans-serif;color:#64748b}}
@page wide{size:A4 landscape;margin:15mm 17mm}
@page tall{size:A3 portrait;margin:16mm 17mm}
body{font:10pt/1.65 "DejaVu Sans",CJK,sans-serif;color:#1e293b}
h1,h2,h3{color:#132b46;line-height:1.4;font-weight:normal}
h1{font-size:24pt;margin:0 0 10mm}
h2{font-size:16pt;margin:8mm 0 4mm;break-after:avoid;bookmark-level:1}
h3{font-size:12pt;margin:6mm 0 3mm;break-after:avoid;bookmark-level:2}
p{margin:0 0 3mm;orphans:3;widows:3}
a{color:#1f5b99;text-decoration:none;overflow-wrap:anywhere}
ul,ol{padding-left:5mm}li{margin:1.5mm 0}
table{width:100%;border-collapse:collapse;margin:4mm 0 6mm;font-size:8.5pt;line-height:1.5}
thead{display:table-header-group}tr{break-inside:avoid}
th,td{padding:2mm;border-bottom:0.2mm solid #dce3ed;vertical-align:top;overflow-wrap:anywhere}
th{background:#edf2f7;text-align:left;font-weight:normal}
code{font:8.5pt "DejaVu Sans Mono",CJK;overflow-wrap:anywhere}
.cover{break-after:page;padding-top:17mm}
.cover .subtitle{color:#52637a;font-size:11pt;margin-bottom:12mm}
.cover h2{bookmark-level:none;font-size:15pt}
.cover ol{list-style:none;padding:0}
.cover li{padding:2mm 0;border-bottom:0.2mm solid #e2e8f0}
.cover a::after{content:leader('.') target-counter(attr(href),page);color:#64748b}
.cover .note{margin-top:12mm;font-size:9pt;color:#52637a}
.plot-page{break-before:page;break-after:page}
.plot-page h2{margin:0 0 3mm;font-size:16pt}
.plot-page h3{margin:0 0 2mm;font-size:11pt}
.filename{font:7.5pt "DejaVu Sans Mono";color:#64748b;margin-bottom:3mm}
.figure-note{font-size:8.5pt;line-height:1.45;margin-bottom:2mm}
.figure-note p{margin:0 0 1mm}
.wide{page:wide}.wide img{display:block;max-width:263mm;max-height:132mm;width:auto;height:auto;margin:4mm auto 0}
.tall{page:tall}.tall img{display:block;max-width:259mm;max-height:345mm;width:auto;height:auto;margin:2mm auto 0}
'''
document=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{css}</style></head><body>
<section class="cover"><h1>{html.escape(title)}</h1><p class="subtitle">120 条轨迹 · 2326 个所选观测 · 68 张静态图</p><h2>目录</h2><ol>{nav}</ol><p class="note">正文、表格和全部静态图已收入本文件。PCA 图集中在末尾附录；可通过目录或 PDF 书签跳转。CSV 数值表与交互 HTML 保留在原报告目录。</p></section>{main}{''.join(chapters)}</body></html>'''
(WORK/'report_print.html').write_text(document)
if '--validate-only' not in sys.argv:
    HTML(string=document,base_url=ROOT.as_uri()+'/').write_pdf(OUTPUT)
pdf=PdfReader(OUTPUT)
texts=[page.extract_text() or '' for page in pdf.pages]
text='\n'.join(texts)
assert '首轮主要观察' in text and '所选观测的阶段支持量' in text
assert '99.5%' in text and '68' in text
def painted_images(stream,resources):
    result=[]
    objects=resources.get('/XObject',{})
    for operands,operation in ContentStream(stream,pdf).operations:
        if operation!=b'Do':continue
        reference=objects[operands[0]]
        value=reference.get_object()
        if value['/Subtype']=='/Image':result.append(str(operands[0]))
        elif value['/Subtype']=='/Form':
            result.extend(painted_images(value,value.get('/Resources',resources)))
    return result
per_page=[painted_images(page.get_contents(),page['/Resources']) for page in pdf.pages]
images=sum(map(len,per_page))
assert images==68,images
assert all(len(used)<=1 for used in per_page)
assert all(re.search(r'图\s*\d{2}',texts[i]) for i,used in enumerate(per_page) if used)
assert '\ufffd' not in text
receipt=dict(passed=True,source=str(SOURCE),pdf=str(OUTPUT),pages=len(pdf.pages),embedded_figures=images,
    size_bytes=OUTPUT.stat().st_size,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    pdf_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    chinese_text_extraction_passed=True,includes_pca_appendix=True,
    figure_pages=[i+1 for i,used in enumerate(per_page) if used])
(WORK/'pdf_validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False),flush=True)
