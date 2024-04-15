def encipher(patient_id: str, key: str):
    output = []
    non_digit_char_count = 0

    for c_idx, char in enumerate(patient_id):
        if not str.isdigit(char):
            output.append(char)
            non_digit_char_count += 1
            continue
        key_idx = c_idx - non_digit_char_count
        k = int(key[key_idx])
        out_char = (int(char) + k) % 10
        output.append(str(out_char))
    
    return f"ANON{''.join(output)}"

def decipher(anon_id: str, key: str):
    output = []
    non_digit_char_count = 0

    for c_idx, char in enumerate(anon_id[4:]):
        if not str.isdigit(char):
            output.append(char)
            non_digit_char_count += 1
            continue
        key_idx = c_idx - non_digit_char_count
        k = int(key[key_idx])
        out_char = (int(char) - k) % 10
        output.append(str(out_char))
    
    return ''.join(output)