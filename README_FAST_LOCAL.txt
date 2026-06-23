FAST LOCAL OCR VERSION

This version is optimized for running on your local PC.

What changed:
- Multiple PDFs can be processed at the same time.
- BACKGROUND_WORKERS controls how many OCR workers run.
- Default local setting is BACKGROUND_WORKERS=3.
- Render should still use BACKGROUND_WORKERS=1.
- PDF text is parsed first; OCR only runs when parsing does not find usable text.
- Page 1 is skipped. The scanner starts at page 2.
- Quick Business Lookup is included.

Recommended local settings:
- Start with BACKGROUND_WORKERS=2.
- If your PC is still responsive, try BACKGROUND_WORKERS=3.
- Only use BACKGROUND_WORKERS=4 if you have a strong CPU and 16GB+ RAM.

How to run locally:
1. Install dependencies:
   pip install -r requirements.txt

2. Make sure Tesseract OCR is installed and added to PATH.

3. Run:
   run_local_fast.bat

For Render:
Set environment variable:
   BACKGROUND_WORKERS=1

Do not set workers too high. OCR is CPU-heavy and can freeze slower computers.
