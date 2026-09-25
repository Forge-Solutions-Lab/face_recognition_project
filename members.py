MEMBERS = [
    {"id": "6601001", "name": "สมชาย", "name_en": "Somchai"},
    {"id": "6601002", "name": "สมหญิง", "name_en": "Somying"},
]
def find_member(member_id):
    return next((m for m in MEMBERS if m["id"] == member_id), None)
def member_dir_name(member):
    return f"{member['id']}_{member['name'].replace(' ', '_')}"
