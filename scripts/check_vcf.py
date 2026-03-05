# end-to-end test for the VCF upload feature. uploads the sample VCF and prints
# everything that comes back. the multipart boundary stuff is ugly but it works.
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import urllib.request
import json

BASE = "http://localhost:8000"

# Health check
resp = urllib.request.urlopen(BASE + "/")
health = json.loads(resp.read())
print("Health:", health)
print(f"  VCF enabled: {health.get('vcf', False)}")
print()

# Upload test VCF
with open("test_variants.vcf", "rb") as f:
    vcf_content = f.read()

# Multipart upload
import http.client
import mimetypes

boundary = "----SteppeDNATestBoundary"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="test_variants.vcf"\r\n'
    f"Content-Type: text/plain\r\n"
    f"\r\n"
).encode() + vcf_content + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    f"{BASE}/predict/vcf",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST"
)

resp = urllib.request.urlopen(req)
result = json.loads(resp.read())

print("=== VCF RESULTS ===")
print(f"  Total variants in file: {result['total_variants_in_file']}")
print(f"  BRCA2 missense found:  {result['brca2_missense_found']}")
print(f"  Skipped:               {result['skipped_count']}")
print()

if result['predictions']:
    print("  Predictions:")
    for p in result['predictions']:
        print(f"    {p['hgvs_p']:20s} cDNA={p['cdna_pos']:5d}  {p['mutation']}  => {p['prediction']:12s} (p={p['probability']:.4f})")
else:
    print("  No predictions returned!")

print()
print("  Skipped reasons:")
for s in result.get('skipped_reasons', []):
    print(f"    Line {s.get('line','?')}: {s.get('reason','unknown')} (pos={s.get('pos','')})")
