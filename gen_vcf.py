import sys
import os
import random

# Add backend directory to sys.path so we can import get_gene_data
sys.path.append('c:/Users/User/OneDrive/Desktop/SteppeDNA/backend')
from main import get_gene_data

genes = ['BRCA1', 'BRCA2', 'PALB2', 'RAD51C', 'RAD51D']
chromosomes = {'BRCA1':'17', 'BRCA2':'13', 'PALB2':'16', 'RAD51C':'17', 'RAD51D':'17'} # some actual chroms

out_file = 'c:/Users/User/OneDrive/Desktop/SteppeDNA/comprehensive_test_suite.vcf'
vcf_lines = [
    '##fileformat=VCFv4.2',
    '##source=SteppeDNATestSuite',
    '##INFO=<ID=TEST,Number=1,Type=String,Description="Test Case">',
    '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO'
]

random.seed(42)
bases = ['A', 'C', 'G', 'T']
total_variants = 0

for gene in genes:
    # Get the official data context built by the backend
    gene_data = get_gene_data(gene)
    
    chrom = chromosomes.get(gene, '13')
    
    genomic_to_cdna = gene_data.get('genomic_to_cdna', {})
    if not genomic_to_cdna:
        continue
        
    genomic_positions = list(genomic_to_cdna.keys())
    
    # 1. Generate Missense/Nonsense variants (single nucleotide)
    for i in range(25):
        pos = random.choice(genomic_positions)
        ref = random.choice(bases)
        alt = random.choice([b for b in bases if b != ref])
        vcf_lines.append(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t100\tPASS\tTEST={gene}_MissenseOrNonsense_{i}")
        total_variants += 1
        
    # 2. Generate Frameshift (Indels)
    for i in range(5):
        pos = random.choice(genomic_positions)
        ref = random.choice(bases)
        alt = ref + random.choice(bases) # insertion
        vcf_lines.append(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t100\tPASS\tTEST={gene}_FrameshiftIns_{i}")
        
        pos2 = random.choice(genomic_positions)
        ref2 = random.choice(bases) + random.choice(bases)
        alt2 = ref2[0] # deletion
        vcf_lines.append(f"{chrom}\t{pos2}\t.\t{ref2}\t{alt2}\t100\tPASS\tTEST={gene}_FrameshiftDel_{i}")
        total_variants += 2

    # 3. Multiallelic
    for i in range(3):
        pos = random.choice(genomic_positions)
        ref = 'A'
        alt = 'C,G,T'
        vcf_lines.append(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t100\tPASS\tTEST={gene}_Multiallelic_{i}")
        total_variants += 1
        
    # 4. Out of bounds (introns or outside CDS)
    max_pos = max(map(int, genomic_positions))
    vcf_lines.append(f"{chrom}\t{max_pos + 1000}\t.\tA\tC\t100\tPASS\tTEST={gene}_OutOfBounds_Intron")
    total_variants += 1
    
    # 5. Invalid Alleles
    pos = random.choice(genomic_positions)
    vcf_lines.append(f"{chrom}\t{pos}\t.\tA\tZ\t100\tPASS\tTEST={gene}_InvalidAllele")
    vcf_lines.append(f"{chrom}\t{pos}\t.\tN\tA\t100\tPASS\tTEST={gene}_InvalidRef")
    total_variants += 2
    
    # 6. Wrong chromosome
    vcf_lines.append(f"chr99\t{pos}\t.\tA\tC\t100\tPASS\tTEST={gene}_WrongChrom")
    total_variants += 1
    
with open(out_file, 'w') as f:
    f.write('\n'.join(vcf_lines) + '\n')

print(f"Successfully generated highly varied VCF with {total_variants} variants at {out_file}")
