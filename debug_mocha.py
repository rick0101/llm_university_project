from paperhunter.ingestion.grobid_client import GrobidClient
from paperhunter.models import Paper
from paperhunter.ingestion.bibtex_generator import generate_bibtex_entry

def test():
    pdf_path = "scaricati/Adversarial_Code1/MOCHA_Are_Code_Language_Models_Robust_Against_Multi-Turn_Malicious\n__Coding_Prompts.pdf"
    
    gc = GrobidClient()
    out = gc.process_pdf(pdf_path)
    paper = Paper(
        filename="MOCHA",
        hash="123",
        title=out.title,
        authors=out.authors,
        abstract=out.abstract
    )
    print("GROBID AUTHORS LEN:", len(paper.authors))
    print("GROBID AUTHORS:", paper.authors)
    print("BIBTEX ENTRY:\n", generate_bibtex_entry(paper))

if __name__ == "__main__":
    test()
