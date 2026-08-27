from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(".")
OUT = ROOT / "outputs" / "freesasa27_context_evidence_update_2026_08_25"
SENS = ROOT / "outputs" / "validated_endpoint_sensitivity_2026_08_25"
PANELS = OUT / "structure_panels"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

def fnt(size, bold=False): return ImageFont.truetype(FONT_B if bold else FONT, size)

def center(draw, box, text, font, fill="#111", spacing=5):
    x0,y0,x1,y1=box; bb=draw.multiline_textbbox((0,0),text,font=font,spacing=spacing,align="center"); w,h=bb[2]-bb[0],bb[3]-bb[1]
    draw.multiline_text(((x0+x1-w)/2,(y0+y1-h)/2-bb[1]),text,font=font,fill=fill,spacing=spacing,align="center")

def fit_image(img, box):
    x0,y0,x1,y1=box; copy=img.copy(); copy.thumbnail((x1-x0,y1-y0),Image.Resampling.LANCZOS)
    return copy, (int((x0+x1-copy.width)/2),int((y0+y1-copy.height)/2))

def main():
    usp=Image.open(PANELS/"USP_A_C116_C149_buried.png").convert("RGBA")
    sam=Image.open(PANELS/"SAM_CXXC_vs_CM.png").convert("RGBA")
    recall=pd.read_csv(SENS/"validated_case_endpoint_matrix.csv")
    group=pd.read_csv(SENS/"threshold_sensitivity_group_comparisons_freesasa.csv")
    W,H=5200,3700; im=Image.new("RGBA",(W,H),"white"); d=ImageDraw.Draw(im)

    cards=[(120,120,2540,2020),(2660,120,5080,2020)]
    for box in cards: d.rounded_rectangle(box,radius=30,fill="#FBFCFD",outline="#CCD5DF",width=5)
    d.text((160,150),"A",font=fnt(70,True),fill="#111"); d.text((2700,150),"B",font=fnt(70,True),fill="#111")
    d.text((300,155),"USP-A: geometry present, apo-state exposure absent",font=fnt(42,True),fill="#111")
    d.text((2840,155),"SAM synthase: motif present, motif geometry absent",font=fnt(42,True),fill="#111")
    u,upos=fit_image(usp,(190,330,2470,1560)); im.alpha_composite(u,upos)
    s,spos=fit_image(sam,(2730,330,5010,1560)); im.alpha_composite(s,spos)

    # Residue legend and metric strips
    d.rounded_rectangle((250,1600,2410,1940),radius=22,fill="#FFF5E8",outline="#D9902F",width=4)
    center(d,(280,1620,2380,1740),"C116–C149 = 4.60 Å; recurrent in 3/3 models",fnt(37,True))
    center(d,(280,1740,2380,1860),"median pLDDT 95.4; pair PAE 1.17 Å",fnt(32))
    center(d,(280,1840,2380,1930),"FreeSASA (1.4 Å probe): mean pair 0.13 Å²  →  fails exposure gate",fnt(32,True),fill="#A23B3B")
    d.rounded_rectangle((2790,1600,4950,1940),radius=22,fill="#F7F0FF",outline="#8B6FB1",width=4)
    center(d,(2820,1620,4920,1730),"Conserved C44XXC47: SG–SG = 10.47 Å",fnt(37,True))
    center(d,(2820,1720,4920,1830),"C47–M54 = 4.27 Å; recurrent in 3/3 models",fnt(32))
    center(d,(2820,1820,4920,1930),"mean FreeSASA 19.0 Å², but sequence separation = 7  →  excluded by frozen rule",fnt(30,True),fill="#6A4695")

    # Lower-left symmetric evidence table
    d.rounded_rectangle((120,2120,3060,3560),radius=30,fill="#FFFFFF",outline="#CCD5DF",width=5)
    d.text((160,2160),"C",font=fnt(70,True),fill="#111"); d.text((300,2170),"Two independent failure modes of a rigid structural endpoint",font=fnt(40,True),fill="#111")
    x=[190,1050,2040,2990]; y0=2330; rh=190
    headers=["Feature","USP-A","SAM synthase"]
    for i in range(3):
        d.rectangle((x[i],y0,x[i+1],y0+rh),fill="#324A5F",outline="white",width=3); center(d,(x[i],y0,x[i+1],y0+rh),headers[i],fnt(32,True),fill="white")
    rows=[
        ("Experimental anchor","BiFC + Cu binding/transfer","BiFC only"),
        ("Sequence motif","No CXXC requirement","Conserved C44XXC47"),
        ("Relevant geometry","C116–C149 = 4.60 Å","C44–C47 = 10.47 Å"),
        ("Apo-state exposure","Mean pair SASA 0.13 Å²","Motif mean SASA ~10 Å²"),
        ("Frozen endpoint result","Missed Cu-transfer-positive protein: SASA gate","Missed BiFC-associated protein: geometry / separation"),
    ]
    for r,row in enumerate(rows,1):
        yy=y0+r*rh; fill="#F5F7FA" if r%2 else "#FFFFFF"
        for i,text in enumerate(row):
            d.rectangle((x[i],yy,x[i+1],yy+rh),fill=fill,outline="#D5DCE3",width=3); center(d,(x[i]+10,yy,x[i+1]-10,yy+rh),text,fnt(28, i==0))

    # Lower-right recall heatmap and null group comparisons
    d.rounded_rectangle((3160,2120,5080,3560),radius=30,fill="#FFFFFF",outline="#CCD5DF",width=5)
    d.text((3200,2160),"D",font=fnt(70,True),fill="#111"); d.text((3340,2170),"Rule-family sensitivity and group comparison",font=fnt(39,True),fill="#111")
    endpoint_label={"Frozen":"Frozen","No_SASA_gate":"No SASA","Separation_ge3":"sep ≥3","No_SASA_and_separation_ge3":"No SASA + sep ≥3"}
    cases=["USP-A","SAM synthase","Peroxiredoxin","PR/NTF2-like"]
    eps=list(endpoint_label); hx=[3230,3870,4140,4410,4680,5030]; hy=2350; hh=125
    d.rectangle((hx[0],hy,hx[1],hy+hh),fill="#324A5F")
    center(d,(hx[0],hy,hx[1],hy+hh),"Endpoint",fnt(27,True),fill="white")
    for i,c in enumerate(cases): d.rectangle((hx[i+1],hy,hx[i+2],hy+hh),fill="#324A5F"); center(d,(hx[i+1],hy,hx[i+2],hy+hh),c.replace(" synthase","\nsynthase").replace("/","/\n"),fnt(23,True),fill="white")
    for r,ep in enumerate(eps,1):
        yy=hy+r*hh; d.rectangle((hx[0],yy,hx[1],yy+hh),fill="#F0F3F6",outline="#D5DCE3",width=2); center(d,(hx[0],yy,hx[1],yy+hh),endpoint_label[ep],fnt(25,True))
        for i,c in enumerate(cases):
            val=bool(recall[(recall.endpoint==ep)&(recall.case==c)].endpoint_positive.iloc[0]); fc="#3BA272" if val else "#E56B6F"
            d.rectangle((hx[i+1],yy,hx[i+2],yy+hh),fill=fc,outline="white",width=3); center(d,(hx[i+1],yy,hx[i+2],yy+hh),"+" if val else "—",fnt(38,True),fill="white")
    table_y=2960
    d.text((3240,table_y),"Candidate vs matched background remains null across the rule family",font=fnt(29,True),fill="#333")
    yy=table_y+90
    for ep in eps:
        row=group[group.endpoint==ep].iloc[0]; txt=f"{endpoint_label[ep]:<18}  RR {row.risk_ratio:.2f} ({row.risk_ratio_ci_low:.2f}–{row.risk_ratio_ci_high:.2f}); Fisher P={row.fisher_exact_p:.3f}"
        d.text((3260,yy),txt,font=fnt(26),fill="#333"); yy+=92

    center(d,(180,3580,5020,3680),"The endpoint is useful for calibrated group comparison, but not a sensitive stand-alone detector of experimentally associated or transferred Cu-client sites.",fnt(31,True),fill="#333")
    rgb=im.convert("RGB"); rgb.save(OUT/"Figure_validated_anchor_sensitivity_structural_symmetry.png",dpi=(350,350)); rgb.save(OUT/"Figure_validated_anchor_sensitivity_structural_symmetry.pdf",resolution=350)

    source_rows=[
        {"case":"USP-A","experimental_evidence":"BiFC; Cu binding and contact-dependent Cu transfer","site":"C116-C149","motif":"none required","distance_A":4.60,"median_pair_pae_A":1.165,"median_min_plddt":95.38,"mean_pair_freesasa_A2":0.130,"frozen_failure":"SASA gate"},
        {"case":"SAM synthase motif","experimental_evidence":"BiFC only","site":"C44-C47","motif":"CXXC","distance_A":10.47,"median_pair_pae_A":0.89,"median_min_plddt":98.62,"mean_pair_freesasa_A2":10.1,"frozen_failure":"distance geometry"},
        {"case":"SAM synthase alternative","experimental_evidence":"BiFC only; no Cu transfer shown","site":"C47-M54","motif":"none","distance_A":4.27,"median_pair_pae_A":0.89,"median_min_plddt":98.62,"mean_pair_freesasa_A2":19.0,"frozen_failure":"sequence separation 7"},
    ]
    pd.DataFrame(source_rows).to_csv(OUT/"Figure_validated_anchor_sensitivity_structural_symmetry_source.csv",index=False)

if __name__ == "__main__": main()
