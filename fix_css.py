import re

file_path = 'c:/Users/User/OneDrive/Desktop/SteppeDNA/frontend/index.html'
with open(file_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Replace everything from <style> to </style> with the link tag
pattern = re.compile(r'<style>.*?</style>', re.DOTALL)
new_html = pattern.sub('<link rel="stylesheet" href="styles.css?v=3">', html)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_html)

print('Successfully removed inline styles and linked external CSS!')
