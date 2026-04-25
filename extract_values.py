import re

# OCR text (paste your extracted text here)
text = """
VOLUME 2 ml
SPERM COUNT 17 millions/ml
NORMAL FORMS 08%
Age 33
"""

# Extract values using regex
volume = re.search(r'VOLUME\s*(\d+\.?\d*)', text)
sperm_count = re.search(r'SPERM COUNT\s*(\d+)', text)
morphology = re.search(r'NORMAL FORMS\s*(\d+)', text)
age = re.search(r'Age\s*(\d+)', text)

volume = float(volume.group(1)) if volume else None
sperm_count = float(sperm_count.group(1)) if sperm_count else None
morphology = float(morphology.group(1)) if morphology else None
age = int(age.group(1)) if age else None

print("Extracted Values")
print("Age:", age)
print("Volume:", volume)
print("Sperm Count:", sperm_count)
print("Morphology Normal:", morphology)

