import json
import os
from config import MEMBERS_PATH

def load_members(path=MEMBERS_PATH):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_members(members, path=MEMBERS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(members, f, ensure_ascii=False, indent=2)

def find_member(members, member_id):
    return next((m for m in members if m["id"] == member_id), None)

def upsert_member(members, member):
    return [m for m in members if m["id"] != member["id"]] + [member]

def register_member(member, path=MEMBERS_PATH):
    save_members(upsert_member(load_members(path), member), path)

def member_dir_name(member):
    return f"{member['id']}_{member['name'].replace(' ', '_')}"
