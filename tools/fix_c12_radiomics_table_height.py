from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


BASE = Path(r"C:\Users\Roberto\OneDrive - artelnics.com\DIPCAN\subsanacion_tecnica")
DOCX = BASE / "Anexo C12. Desarrollo y validación clínica de algoritmos_NUEVO.docx"


def main():
    doc = Document(DOCX)
    table = next(
        table
        for table in doc.tables
        if any("1.018" in cell.text for row in table.rows for cell in row.cells)
    )
    for row in table.rows:
        tr_pr = row._tr.get_or_add_trPr()
        for tr_height in list(tr_pr.findall(qn("w:trHeight"))):
            tr_pr.remove(tr_height)
    doc.save(DOCX)
    print(DOCX)


if __name__ == "__main__":
    main()
