import mPDF
import optparse
import sys

def Main():
    parser = optparse.OptionParser(usage='usage: %prog [options] pdf-file', version='%prog 0.1')
    parser.add_option('-j', '--javascript', help='javascript to embed (default embedded JavaScript is app.alert messagebox)')
    parser.add_option('-f', '--javascriptfile', help='javascript file to embed (default embedded JavaScript is app.alert messagebox)')
    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.print_help()
        print('')
        print('  make-pdf-javascript, use it to create a PDF document with embedded JavaScript that will execute automatically when the document is opened')
        print('  Source code put in the public domain by Didier Stevens, no Copyright')
        print('  Use at your own risk')
        print('  https://DidierStevens.com')
    
    else:
        # Note: Ensure mPDF.py is in the same folder and updated for Python 3
        oPDF = mPDF.cPDF(args[0])
    
        oPDF.header()
    
        # PDF Structure: /OpenAction points to object 7 (the JavaScript)
        oPDF.indirectobject(1, 0, '<<\n /Type /Catalog\n /Outlines 2 0 R\n /Pages 3 0 R\n /OpenAction 7 0 R\n>>')
        oPDF.indirectobject(2, 0, '<<\n /Type /Outlines\n /Count 0\n>>')
        oPDF.indirectobject(3, 0, '<<\n /Type /Pages\n /Kids [4 0 R]\n /Count 1\n>>')
        oPDF.indirectobject(4, 0, '<<\n /Type /Page\n /Parent 3 0 R\n /MediaBox [0 0 612 792]\n /Contents 5 0 R\n /Resources <<\n             /ProcSet [/PDF /Text]\n             /Font << /F1 6 0 R >>\n            >>\n>>')
        oPDF.stream(5, 0, 'BT /F1 12 Tf 100 700 Td 15 TL (JavaScript example) Tj ET')
        oPDF.indirectobject(6, 0, '<<\n /Type /Font\n /Subtype /Type1\n /Name /F1\n /BaseFont /Helvetica\n /Encoding /MacRomanEncoding\n>>')
    
        if options.javascript == None and options.javascriptfile == None:
            javascript = """app.alert({cMsg: 'Hello from PDF JavaScript', cTitle: 'Testing PDF JavaScript', nIcon: 3});"""
        elif options.javascript != None:
            javascript = options.javascript
        else:
            try:
                # Open as text in Python 3 for easier formatting into the PDF string
                with open(options.javascriptfile, 'r', encoding='utf-8', errors='ignore') as fileJavaScript:
                    javascript = fileJavaScript.read()
            except Exception as e:
                print(f"error reading file {options.javascriptfile}: {e}")
                return
        
        # Object 7: The JavaScript Action
        oPDF.indirectobject(7, 0, '<<\n /Type /Action\n /S /JavaScript\n /JS (%s)\n>>' % javascript)
    
        oPDF.xrefAndTrailer('1 0 R')

if __name__ == '__main__':
    Main()