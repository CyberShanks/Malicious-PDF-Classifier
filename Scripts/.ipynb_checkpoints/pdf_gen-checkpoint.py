import pikepdf
from pikepdf import NameTree # Import the low-level NameTree tool

# 1. Load your realistic base template
pdf = pikepdf.Pdf.open("../Resources/InvoiceTemplate.pdf")

# 2. Read the binary data of your "payload"
with open("../Resources/payload.exe", "rb") as f:
    payload_data = f.read()

# 2. INJECT FIRST (Standard Injection)
# We let pikepdf handle the complex job of creating the Names/EmbeddedFiles structure.
pdf.attachments['malicious_update.exe'] = payload_data

# 3. EDIT LATER (Low-Level Modification)
# Now we go "under the hood" to find the object we just created.
# We access the raw NameTree where embedded files live.
names = NameTree(pdf.Root.Names.EmbeddedFiles)

# We retrieve the raw FileSpec dictionary (not the wrapper!)
file_spec = names['malicious_update.exe']

# Access the Embedded File Stream (The 'EF' dictionary -> 'F' key)
stream = file_spec.EF.F

# 4. Inject the Metadata
# Now we can modify the raw stream object directly.
stream.Params = pikepdf.Dictionary(
    Size=len(payload_data),
    CreationDate=pikepdf.String("D:20230101000000Z"),
    ModDate=pikepdf.String("D:20230101000000Z"),
    CheckSum=pikepdf.String("1234567890ABCDEF") # Fake checksum common in malware
)

# 5. Save
pdf.save("test_malicious_embedded.pdf")
print("Success: Malicious file created with custom 'Params' metadata.")