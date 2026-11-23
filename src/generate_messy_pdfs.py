"""Generate receipt-style messy medical claim PDFs for the Claim Intake demo.

Writes data/messy/sample_01_clinic_receipt.pdf and sample_02_pt_superbill.pdf.
The PDFs hold real (extractable) text, so the rule-based intake parser can read them.
They deliberately contain messy details: mixed date formats, IDs without dashes,
a typo'd reference number, a missing price, totals that don't add up, handwritten notes.
"""
from pathlib import Path

from reportlab.lib.colors import HexColor, black
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "data" / "messy"
INK = HexColor("#1f3fa8")      # pen-blue for "handwritten" notes
STAMP = HexColor("#b03a2e")
GREY = HexColor("#555555")


def dashed(c, x1, x2, y):
    c.setDash(2, 2)
    c.line(x1, y, x2, y)
    c.setDash()


def clinic_receipt(path):
    """Thermal-printer style patient receipt (80 mm wide)."""
    w, h = 80 * mm, 168 * mm
    c = canvas.Canvas(str(path), pagesize=(w, h))
    c.setTitle("Moreno Family Clinic - patient receipt")
    L, R = 5 * mm, w - 5 * mm
    y = h - 12 * mm

    def line(txt, font="Courier", size=8, center=False, right=None, dy=4 * mm):
        nonlocal y
        c.setFont(font, size)
        if center:
            c.drawCentredString(w / 2, y, txt)
        else:
            c.drawString(L, y, txt)
        if right is not None:
            c.drawRightString(R, y, right)
        y -= dy

    line("MORENO FAMILY CLINIC", "Courier-Bold", 11, center=True, dy=4.5 * mm)
    line("1180 Bayview Ave, Suite 4", center=True)
    line("Tel (555) 014-2290", center=True)
    line("PATIENT RECEIPT / CLAIM COPY", "Courier-Bold", 8, center=True, dy=5 * mm)
    dashed(c, L, R, y + 2 * mm); y -= 2 * mm
    line("Rcpt#: 00-4471        11/3/26 10:42")
    line("Patient : ALVAREZ, J.")
    line("Member #: MBR101082")
    line("Provider: Dr B. Moreno (Fam Med)")
    line("Prov ID : PRV-1045")
    line("Orig ref: CLM-10O417", dy=5 * mm)
    dashed(c, L, R, y + 2 * mm); y -= 2 * mm
    line("CODE   DESCRIPTION      QTY  AMOUNT", "Courier-Bold", 7.5)
    line("99215  OFFICE VST EST     1", right="225.67")
    line("       HIGH CMPLX", size=7)
    line("96127  BRIEF EMOT/BEHAV   3", right="868.83")
    line("       ASSESS @289.61", size=7)
    line("Dx: z00.00 GENERAL EXAM", size=7.5, dy=5 * mm)
    dashed(c, L, R, y + 2 * mm); y -= 2 * mm
    line("SUBTOTAL", right="1,094.50")
    line("PT COPAY PAID", right="-25.00")
    line("BALANCE TO INSURER", "Courier-Bold", 8.5, right="1,069.50", dy=6 * mm)
    line("PAID: VISA ****  COPAY ONLY", size=7)
    line("Submitted to ins: 11/17/2026", size=7, dy=6 * mm)
    line("** THANK YOU - KEEP FOR RECORDS **", size=7, center=True, dy=8 * mm)

    # handwritten-style note from billing clerk
    c.setFillColor(INK)
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(L, y, "resubmit - portal timed out")
    c.drawString(L, y - 4.5 * mm, "ref # typo?? pls check  - Rosa")
    # RESUBMITTED stamp
    c.saveState()
    c.setStrokeColor(STAMP); c.setFillColor(STAMP); c.setLineWidth(1.4)
    c.translate(w / 2 + 6 * mm, 22 * mm); c.rotate(18)
    c.roundRect(-24 * mm, -5 * mm, 48 * mm, 10 * mm, 2 * mm)
    c.setFont("Helvetica-Bold", 12); c.drawCentredString(0, -1.6 * mm, "RESUBMITTED")
    c.restoreState()
    c.showPage(); c.save()


def pt_superbill(path):
    """Faxed physical-therapy superbill with a code table and handwritten edits."""
    w, h = 216 * mm, 279 * mm
    c = canvas.Canvas(str(path), pagesize=(w, h))
    c.setTitle("Harbor PT - superbill")
    L, R = 18 * mm, w - 18 * mm

    c.setFont("Courier", 7); c.setFillColor(GREY)
    c.drawString(L, h - 8 * mm, "FAX  Oct-30-2026 16:07   FROM: 555 014 7781   TO: CLAIMS INTAKE   P.01/01")
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 16); c.drawString(L, h - 22 * mm, "HARBOR PHYSICAL THERAPY & REHAB")
    c.setFont("Helvetica", 9)
    c.drawString(L, h - 28 * mm, "Superbill / Daily Treatment Log        Provider: PRV 1011   (NPI on file)")
    c.line(L, h - 31 * mm, R, h - 31 * mm)

    y = h - 40 * mm
    c.setFont("Helvetica", 10)
    for lbl, val in [("Patient", "WEBB, T."), ("Member ID", "mbr-101794"),
                     ("Date of service", "2026-10-14"), ("Therapist", "Dr G. Walsh, DPT")]:
        c.setFont("Helvetica-Bold", 10); c.drawString(L, y, f"{lbl}:")
        c.setFont("Helvetica", 10); c.drawString(L + 34 * mm, y, val)
        y -= 6 * mm
    c.setFont("Helvetica-Bold", 10); c.drawString(L, y, "Diagnosis:")
    c.setFont("Helvetica", 10); c.drawString(L + 34 * mm, y, "M25.561  pain in R knee")
    c.setFillColor(INK); c.setFont("Helvetica-Oblique", 10)
    c.drawString(L + 85 * mm, y, "<- chart says L knee??")
    c.setFillColor(black)

    # code table
    y -= 12 * mm
    cols = [L, L + 12 * mm, L + 32 * mm, L + 110 * mm, L + 132 * mm, R]
    heads = ["", "CPT", "Description", "Units", "Rate ea", ""]
    c.setFillColor(HexColor("#e6e6e6")); c.rect(L, y - 2 * mm, R - L, 7 * mm, fill=1, stroke=0)
    c.setFillColor(black); c.setFont("Helvetica-Bold", 9)
    for x, t in zip(cols, heads):
        c.drawString(x + 2 * mm, y, t)
    rows = [("x", "97110", "Therapeutic exercise (15 min)", "4", "$ 95.00"),
            ("x", "97140", "Manual therapy (15 min)", "4", "$ 88.00"),
            ("x", "97530", "Therapeutic activities (15 min)", "4", "$ 92.50"),
            ("x", "97112", "Neuromuscular re-education (15 min)", "3", ""),
            (" ", "97161", "PT evaluation, low complexity", "", "$ 120.00"),
            (" ", "97035", "Ultrasound (15 min)", "", "$ 40.00")]
    c.setFont("Helvetica", 9)
    for chk, code, desc, units, rate in rows:
        y -= 7 * mm
        c.rect(L + 2 * mm, y - 0.5 * mm, 3.5 * mm, 3.5 * mm)
        if chk == "x":
            c.setFillColor(INK); c.setFont("Helvetica-Bold", 10)
            c.drawString(L + 2.4 * mm, y, "X"); c.setFillColor(black); c.setFont("Helvetica", 9)
        c.drawString(cols[1] + 2 * mm, y, code)
        c.drawString(cols[2] + 2 * mm, y, desc)
        if units:
            c.setFillColor(INK); c.setFont("Helvetica-Oblique", 11)
            c.drawString(cols[3] + 4 * mm, y, units)
            c.setFillColor(black); c.setFont("Helvetica", 9)
        c.drawString(cols[4] + 2 * mm, y, rate)
        c.setStrokeColor(HexColor("#bbbbbb")); c.line(L, y - 2.5 * mm, R, y - 2.5 * mm); c.setStrokeColor(black)

    y -= 14 * mm
    c.setFillColor(INK); c.setFont("Helvetica-Oblique", 11)
    c.drawString(L, y, "Total time approx 3.5 hrs today - pt tolerated well")
    c.drawString(L, y - 6 * mm, "re-ed amount missing, will add later")
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(R, y, "CHARGES TOTAL:  $1,382.00")
    c.setFont("Helvetica", 9)
    c.drawRightString(R, y - 6 * mm, "Date submitted: 10/30/2026")

    y -= 22 * mm
    c.line(L, y, L + 60 * mm, y)
    c.setFont("Helvetica", 8); c.drawString(L, y - 4 * mm, "Provider signature")
    c.setFillColor(INK); c.setFont("Helvetica-Oblique", 16); c.drawString(L + 4 * mm, y + 2 * mm, "GW")
    c.setFillColor(GREY); c.setFont("Courier", 7)
    c.drawString(L, 12 * mm, "This superbill is a statement of services rendered. Please submit to your insurer.")
    c.showPage(); c.save()


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    clinic_receipt(OUT / "sample_01_clinic_receipt.pdf")
    pt_superbill(OUT / "sample_02_pt_superbill.pdf")
    print("wrote", *sorted(p.name for p in OUT.glob("*.pdf")))
