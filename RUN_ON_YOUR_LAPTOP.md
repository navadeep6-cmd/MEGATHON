# How to run the project on your laptop

This takes about 30 minutes for the quick test, and 1–2 hours for the full run, which mostly waits for downloads. You need internet and about 5 GB of free disk space. Keep the laptop plugged in.

## 1. Install Python 3.11 (one time)
- Go to https://www.python.org/downloads/ and download **Python 3.11** (any 3.11.x).
  ⚠️ Python 3.13 or newer will **not** work.
- **Windows:** in the installer, tick **"Add python.exe to PATH"** before clicking Install.
- **Mac:** run the installer, then double-click **"Install Certificates.command"** in the Python 3.11 folder that opens.

## 2. Get the project
Unzip `megathon_code.zip` (double-click it), or on the team's GitHub page click the green **Code** button, then **Download ZIP**, and unzip that.

## 3. Open a terminal inside the project folder
- **Mac:** open the **Terminal** app, type `cd ` (with a space after it), drag the unzipped folder into the Terminal window, then press **Enter**.
- **Windows:** open the unzipped folder in File Explorer, click the address bar, type `cmd`, then press **Enter**.

## 4. Install the libraries (one time, 5–15 min)
Copy and paste these lines one at a time, pressing Enter after each.

**Mac:**
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**Windows:**
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
You will see many download lines. It is fine as long as the last lines do not say `ERROR`.

## 5. Check that everything works
```
python -m unittest discover -s tests
```
✅ The last line must say **OK**.

## 6. Tell Wikimedia who is downloading
Use your GitHub profile link.
- **Mac:** `export COMMONS_CONTACT="https://github.com/YOUR-USERNAME"`
- **Windows:** `set COMMONS_CONTACT=https://github.com/YOUR-USERNAME`

## 7. Quick test run (about 15–30 min)
```
python run_all.py --quick
```
✅ It should end with **Done**.
❌ If it says **FAILED at: …**, copy the red error text and send it to the team.

## 8. Full run (1–2 hours)
```
python run_all.py
```
Results appear in **`reports/RESULTS.md`**. Upload the whole `reports` folder to GitHub with **Add file → Upload files**.
⚠️ Never upload the `data`, `artifacts` or `.venv` folders. They are huge.

## 9. Demo website
```
python app/demo.py
```
Open http://127.0.0.1:7860 in your browser and upload a photo.

**Next time:** if you close the terminal, repeat step 3, then run the activate line from step 4 (`source .venv/bin/activate` on Mac, `.venv\Scripts\activate` on Windows), then step 6, before running anything.
