import asyncio
from paperhunter.ingestion.grobid_client import GrobidClient

async def test():
    client = GrobidClient()
    res = await client.process_pdf("scaricati/Adversarial_Code1/MOCHA_Are_Code_Language_Models_Robust_Against_Multi-Turn_Malicious__Coding_Prompts.pdf")
    print(res.authors)

asyncio.run(test())
