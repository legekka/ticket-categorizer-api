import bleach

with open("formats/default.txt", 'r') as f:
    default_format = f.read()

def clean_text(text):
    if text is None:
        return ""
    return bleach.clean(text, strip=True).replace('&nbsp;', '').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

class Formatter:
    def __init__(self, format_path):
        with open(format_path, 'r') as f:
            self.format = f.read()
            
    def __call__(self, params):
        return self.format.format(**params)