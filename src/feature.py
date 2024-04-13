import re

def API_completion(prompt):
    digit_flag = False
    alpha_flag = False
    underline_flag = False
    for c in prompt[::-1]:
        if c.isdigit():
            digit_flag = True
            continue
        elif c.isalpha():
            alpha_flag = True
            continue
        elif c == "_":
            underline_flag = True
            continue
        else:
            break
    if digit_flag and not alpha_flag and not underline_flag:
        return 0

    if digit_flag or alpha_flag or underline_flag:
        return 1
    return 0

def get_last_line(prompt):
    lines = prompt.split("\n")
    last_line = lines[-1]
    for i, line in enumerate(lines[-2::-1], 1):
        if line.endswith("\\"):
            last_line = line[:-1] + last_line
        else:
            break
    return last_line

def is_inside_string(prompt):
    code = get_last_line(prompt)

    single_quotes_count = code.count("'")
    double_quotes_count = code.count('"')
    back_quotes_count = code.count('`')

    inside_single_quotes = single_quotes_count % 2 == 1
    inside_double_quotes = double_quotes_count % 2 == 1
    inside_back_quotes = back_quotes_count % 2 == 1
    res = inside_single_quotes or inside_double_quotes or inside_back_quotes
    return res

def is_API_invocation_or_completion(prompt):
    # ).
    if prompt[-2:] == ').':
        is_invocation = True

    in_string =  is_inside_string(prompt)
    double_quote_pattern = r'f"[^"]*{[^"]*\.$'
    single_quote_pattern = r"f'[^']*{[^']*\.$"
    is_f_string = False
    if re.findall(double_quote_pattern, prompt) or re.findall(single_quote_pattern, prompt):
        is_f_string = True
    in_string = in_string and not is_f_string
    if in_string:
        return False

    is_invocation = False
    if  prompt[-1] == '.' and API_completion(prompt[:-1]) == 1:
        is_invocation = True

    is_API_completion = bool(API_completion(prompt))

    return (is_invocation or is_API_completion)