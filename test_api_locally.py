import httpx
import asyncio

async def test():
    vcf_content = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
17\t43045678\t.\tG\tA\t.\t.\t.
"""
    files = {'file': ('test.vcf', vcf_content, 'text/plain')}
    data = {'gene': 'BRCA1'}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post('http://localhost:8000/predict/vcf', data=data, files=files, timeout=60.0)
            print(f"Status: {r.status_code}")
            print(f"Response: {r.json()}")
    except Exception as e:
        print(f"Connection failed: {e}")

asyncio.run(test())
