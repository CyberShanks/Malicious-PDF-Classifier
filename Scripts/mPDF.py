# mPDF.py - Python 3 version
# Helper class for Didier Stevens PDF tools
class cPDF:
    def __init__(self, filename):
        self.filename = filename
        self.f = open(filename, 'w', encoding='latin-1')
        self.offset = 0

    def header(self):
        self.write('%PDF-1.1\n')

    def write(self, str):
        self.f.write(str)
        self.offset += len(str)

    def indirectobject(self, index, generation, content):
        self.write(f'{index} {generation} obj\n')
        self.write(content + '\n')
        self.write('endobj\n')

    def stream(self, index, generation, content):
        self.write(f'{index} {generation} obj\n')
        self.write(f'<< /Length {len(content)} >>\n')
        self.write('stream\n')
        self.write(content + '\n')
        self.write('endstream\n')
        self.write('endobj\n')

    def xrefAndTrailer(self, root):
        self.write('xref\n')
        self.write('0 8\n') # Adjust if you have more/fewer objects
        self.write('0000000000 65535 f \n')
        # Simplified for generation - in a real PDF these offsets matter
        self.write('trailer\n')
        self.write(f'<< /Size 8 /Root {root} >>\n')
        self.write('startxref\n')
        self.write('0\n')
        self.write('%%EOF\n')
        self.f.close()
