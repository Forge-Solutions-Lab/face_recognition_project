import json
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox
from config import BASE_DIR
FIELDS = [("id", "รหัสนักศึกษา"), ("name", "ชื่อ (ไทย)"), ("name_en", "ชื่อ (อังกฤษ)")]

def make_window(count):
    root = tk.Tk()
    root.title("ลงทะเบียนสมาชิก")
    root.attributes("-topmost", True)
    tk.Label(root, text=f"ถ่ายไว้ {count} รูป กรอกข้อมูลเจ้าของรูป").grid(row=0, column=0, columnspan=2, pady=8)
    return root

def make_entries(root):
    entries = {}
    for row, (key, text) in enumerate(FIELDS, 1):
        tk.Label(root, text=text).grid(row=row, column=0, padx=8, pady=4, sticky="w")
        entries[key] = tk.Entry(root, width=28)
        entries[key].grid(row=row, column=1, padx=8, pady=4)
    entries["id"].focus_force()
    return entries

def make_buttons(root, on_save):
    tk.Button(root, text="บันทึก", width=10, command=on_save).grid(row=len(FIELDS) + 1, column=0, pady=8)
    tk.Button(root, text="ยกเลิก", width=10, command=root.destroy).grid(row=len(FIELDS) + 1, column=1, pady=8)
    root.bind("<Return>", lambda _: on_save())

def read_entries(entries):
    return {key: " ".join(entry.get().split()) for key, entry in entries.items()}

def validate(member):
    if not all(member.values()):
        return "กรุณากรอกให้ครบทุกช่อง"
    if not member["id"].isdigit():
        return "รหัสนักศึกษาต้องเป็นตัวเลข"
    if not member["name_en"].isascii():
        return "ชื่ออังกฤษต้องพิมพ์เป็นภาษาอังกฤษ"
    return None

def submit(root, entries, result):
    member = read_entries(entries)
    error = validate(member)
    if error:
        return messagebox.showerror("ข้อมูลไม่ถูกต้อง", error, parent=root)
    result.update(member)
    root.destroy()

def ask_member(count):
    result = {}
    root = make_window(count)
    entries = make_entries(root)
    make_buttons(root, lambda: submit(root, entries, result))
    root.mainloop()
    return result or None

def ask_member_process(count):
    cmd = [sys.executable, "-m", "src.form", str(count)]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, cwd=BASE_DIR, check=True).stdout.strip()
    return json.loads(out) if out else None

if __name__ == "__main__":
    member = ask_member(int(sys.argv[1]))
    print(json.dumps(member) if member else "")
