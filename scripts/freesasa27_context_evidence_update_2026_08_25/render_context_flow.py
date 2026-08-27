from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(".")
OUT = ROOT / "outputs" / "freesasa27_context_evidence_update_2026_08_25"
DATA = OUT / "formal27_freesasa_functional_context.csv"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

CONTEXTS = [
    "Canonical/conserved enzyme-fold coincidence", "Native metal/catalytic-site alternative",
    "Scaffold/proteostasis context", "Compartment-incompatible/indirect", "Annotation conflict/QC",
]
CONTEXT_LABEL = {
    CONTEXTS[0]: "Conserved enzyme/fold\ncoincidence", CONTEXTS[1]: "Native metal or\ncatalytic alternative",
    CONTEXTS[2]: "Scaffold /\nproteostasis context", CONTEXTS[3]: "Compartment-incompatible\nor indirect",
    CONTEXTS[4]: "Annotation\nconflict / QC",
}
COLORS = {CONTEXTS[0]: "#4C78A8", CONTEXTS[1]: "#F2A541", CONTEXTS[2]: "#8B6FB1", CONTEXTS[3]: "#D65F5F", CONTEXTS[4]: "#7A8B99"}
DISPOSITIONS = [
    "Secondary interaction candidate (transfer unproven)", "Negative-calibration/native-site benchmark",
    "Exclude from direct-client interpretation", "Background benchmark",
]
DISP_LABEL = {
    DISPOSITIONS[0]: "Secondary interaction\nonly (n=3)", DISPOSITIONS[1]: "Negative/native-site\ncalibration (n=3)",
    DISPOSITIONS[2]: "Exclude direct-client\ninterpretation (n=11)", DISPOSITIONS[3]: "Matched-background\nbenchmark (n=10)",
}

def font(size, bold=False): return ImageFont.truetype(FONT_B if bold else FONT, size)

def center_text(draw, box, text, fnt, fill="#111111", spacing=4):
    x0, y0, x1, y1 = box
    bb = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=spacing, align="center")
    w, h = bb[2]-bb[0], bb[3]-bb[1]
    draw.multiline_text(((x0+x1-w)/2, (y0+y1-h)/2-bb[1]), text, font=fnt, fill=fill, spacing=spacing, align="center")

def intervals(labels, counts, gap, y0, total_height):
    usable = total_height - gap * (len(labels)-1); scale = usable / sum(counts[x] for x in labels)
    out, cursor = {}, y0 + total_height
    for label in labels:
        h = counts[label] * scale; out[label] = (cursor-h, cursor); cursor -= h + gap
    return out, scale

def bezier(p0, p1, p2, p3, t):
    u = 1-t
    return (u**3*p0[0]+3*u*u*t*p1[0]+3*u*t*t*p2[0]+t**3*p3[0], u**3*p0[1]+3*u*u*t*p1[1]+3*u*t*t*p2[1]+t**3*p3[1])

def ribbon(image, x0, a0, b0, x1, a1, b1, color, alpha=100):
    c0, c1 = x0+(x1-x0)*0.42, x0+(x1-x0)*0.58
    top = [bezier((x0,a0),(c0,a0),(c1,a1),(x1,a1),i/30) for i in range(31)]
    bottom = [bezier((x1,b1),(c1,b1),(c0,b0),(x0,b0),i/30) for i in range(31)]
    overlay = Image.new("RGBA", image.size, (0,0,0,0)); od = ImageDraw.Draw(overlay)
    rgb = tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4)); od.polygon(top+bottom, fill=rgb+(alpha,)); image.alpha_composite(overlay)

def main():
    df = pd.read_csv(DATA); W, H = 5200, 3000
    im = Image.new("RGBA", (W, H), "white"); d = ImageDraw.Draw(im)
    d.text((130, 90), "A", font=font(72, True), fill="#111"); d.text((270, 95), "FreeSASA correction of the formal endpoint", font=font(46, True), fill="#111")
    d.text((1500, 90), "B", font=font(72, True), fill="#111"); d.text((1640, 95), "Functional-context triage of 27 corrected positives", font=font(46, True), fill="#111")
    boxes = [
        (230,520,1300,850,"Approximate SASA\n31 positives","#E7EDF5","#4C78A8"),
        (230,1050,1300,1380,"26 retained + 1 added\n5 removed","#E4F3EC","#2E8B57"),
        (230,1580,1300,1910,"Final FreeSASA endpoint\n27 positives","#FFF3D8","#C58A16"),
        (230,2200,1300,2530,"17 candidate\n10 matched background","#F4F1FA","#7B61A8"),
    ]
    for x0,y0,x1,y1,txt,fc,ec in boxes:
        d.rounded_rectangle((x0,y0,x1,y1), radius=24, fill=fc, outline=ec, width=7); center_text(d,(x0,y0,x1,y1),txt,font(42,True))
    for ya,yb in [(850,1050),(1380,1580),(1910,2200)]:
        d.line((765,ya+30,765,yb-45),fill="#4A5568",width=7); d.polygon([(745,yb-65),(785,yb-65),(765,yb-30)],fill="#4A5568")

    x_group=(1700,2070); x_context=(2960,3490); x_disp=(4300,5050); y0,total=430,2370
    groups=["Candidate","Background"]; gc={g:int((df.group==g).sum()) for g in groups}; cc={c:int((df.context_category==c).sum()) for c in CONTEXTS}; dc={q:int((df.disposition_bucket==q).sum()) for q in DISPOSITIONS}
    giv,scale=intervals(groups,gc,110,y0,total); civ,_=intervals(CONTEXTS,cc,55,y0,total); div,_=intervals(DISPOSITIONS,dc,75,y0,total)
    gcur={g:giv[g][1] for g in groups}; clcur={c:civ[c][1] for c in CONTEXTS}
    for g in groups:
        for c in CONTEXTS:
            n=int(((df.group==g)&(df.context_category==c)).sum())
            if n:
                h=n*scale; ga,gb=gcur[g]-h,gcur[g]; ca,cb=clcur[c]-h,clcur[c]; ribbon(im,x_group[1],ga,gb,x_context[0],ca,cb,COLORS[c]); gcur[g]-=h; clcur[c]-=h
    crcur={c:civ[c][1] for c in CONTEXTS}; dcur={q:div[q][1] for q in DISPOSITIONS}
    for c in CONTEXTS:
        for q in DISPOSITIONS:
            n=int(((df.context_category==c)&(df.disposition_bucket==q)).sum())
            if n:
                h=n*scale; ca,cb=crcur[c]-h,crcur[c]; qa,qb=dcur[q]-h,dcur[q]; ribbon(im,x_context[1],ca,cb,x_disp[0],qa,qb,COLORS[c]); crcur[c]-=h; dcur[q]-=h
    d = ImageDraw.Draw(im)
    d.text((1780,300),"Formal group",font=font(34,True),fill="#555"); d.text((3030,300),"Native functional context",font=font(34,True),fill="#555"); d.text((4480,300),"Interpretive disposition",font=font(34,True),fill="#555")
    for g in groups:
        a,b=giv[g]; fc="#EAF2FA" if g=="Candidate" else "#F1F2F4"; d.rounded_rectangle((x_group[0],a,x_group[1],b),radius=18,fill=fc,outline="#34495E",width=5); center_text(d,(x_group[0],a,x_group[1],b),f"{g}\nn={gc[g]}",font(34,True))
    for c in CONTEXTS:
        a,b=civ[c]; d.rounded_rectangle((x_context[0],a,x_context[1],b),radius=16,fill=COLORS[c],outline="white",width=4); center_text(d,(x_context[0],a,x_context[1],b),f"{CONTEXT_LABEL[c]}\n(n={cc[c]})",font(27,True),fill="white")
    for q in DISPOSITIONS:
        a,b=div[q]; d.rounded_rectangle((x_disp[0],a,x_disp[1],b),radius=16,fill="#F8F8F7",outline="#5D6D7E",width=4); center_text(d,(x_disp[0],a,x_disp[1],b),DISP_LABEL[q],font(29,True))
    center_text(d,(140,2820,5060,2960),"Geometry is an opportunity signal: functional context determines follow-up, calibration, or exclusion from direct-client claims.",font(34),fill="#444")
    rgb=im.convert("RGB"); rgb.save(OUT/"Figure_FreeSASA27_functional_context_flow.png",dpi=(350,350)); rgb.save(OUT/"Figure_FreeSASA27_functional_context_flow.pdf",resolution=350)

if __name__ == "__main__": main()
