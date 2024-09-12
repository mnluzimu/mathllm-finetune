import json
import os
import re
from latex2sympy2 import latex2sympy
from tqdm import tqdm
from argparse import ArgumentParser
import sympy as sp
from glob import glob

def save_jsonl(datas, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for data in datas:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def load_jsonl(in_file):
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas

def find_all_numbers(text):
    # last_line = text.split("\n\n")[-1]
    pattern = re.compile('oxed{(.*)}',flags=re.S)
    answers = pattern.findall(text)
    for i in range(len(answers)):
        answers[i] = answers[i].replace(",", "")

    all_numbers = []
    for answer in answers:
        # The regex pattern to match any number (integer or floating-point)
        pattern = r'-?\d+(?:\.\d+)?'
        
        # Using findall to get all occurrences of numbers in the string
        all_numbers.extend(re.findall(pattern, answer))
    
    # If there are no numbers in the string, return None
    if not all_numbers:
        return None
    
    # Return the last number found in the string
    return all_numbers

def delete_extra_zero(n):
    '''删除小数点后多余的0'''
    try:
        n=float(n)
    except:
        print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  # 删除小数点后多余的0
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  # 只剩小数点直接转int，否则转回float
        n=str(n)
        return n


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def convert_division_to_frac(expression):
    def replace_func(match):
        # Extract matched groups
        numerator = match.group(1) or match.group(3)
        denominator = match.group(2) or match.group(4)
        
        # Ensure both numerator and denominator are stripped of surrounding whitespace
        numerator = numerator.strip()
        denominator = denominator.strip()
        
        return f"\\frac{{{numerator}}}{{{denominator}}}"

    # Regular expression to match division patterns
    pattern = r'\(\s*([^()]+?)\s*\)\s*/\s*\(\s*([^()]+?)\s*\)|([^()\s]+)\s*/\s*([^()\s]+)'

    # Replace the division with LaTeX \frac{}{} format
    converted_expression = re.sub(pattern, replace_func, expression)

    return converted_expression


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    splits = string.split("\\text{ ")
    # assert len(splits) == 2
    return splits[-1]


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

    
def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]

    # fix sqrt3 --> sqrt{3}
    if 'sqrt' in string:
        string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    if 'sqrt' in string:
        string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    if "/" in string:
        string = _fix_a_slash_b(string)
        # string = convert_division_to_frac(string)

    return string
    
    
def find_math_answer(s:str):
    s = s.lower()
    s = s.replace('{}','')
    
    tmp = extract_boxed_answer(s)
    if tmp != "":
        ans = tmp
    else:
        ans = s 
 
    if ans.find('}') != -1 and(ans.find('{') ==-1 or  ans.find('}') < ans.find('{')):
        ans=ans.split('}')[0]
    # remove
    if "," in ans:
        ans = ",".join([find_math_answer(e) for e in ans.split(",")])
    ans = ans.split('=')[-1]
    ans = ans.split('\\approx')[-1]
    if ans.endswith(" m"):
        ans = ans[:-2]
    words = ans.split(" ")
    if len(words) == 2 and re.fullmatch(r"\D+", words[-1]) is not None and re.fullmatch(r"[+-]?\d+[\.?\d*]?", words[0]) is not None:
        ans = words[0]
    ans = ans.replace(" ", "")
    ans = ans.replace("\\,", "")
    ans = ans.replace('∞','\\infty').replace("+\infty", "\infty")
    ans = ans.replace("\\\\", "\\")
    ans = ans.replace("\n", "")
    ans = re.sub(r"\\text\{(.*)\}", "", ans)
    ans = ans.replace('\\text', '').replace('\\mbox', '')
    ans = ans.replace('bmatrix', 'pmatrix')
    ans = ans.replace("\\left", "").replace('\\right', '')
    ans = ans.replace("^{\\circ}", "").replace("^\\circ", "")
    ans = ans.replace("{m}^3", "").replace("m^3", "")
    ans = ans.replace("{units}", "").replace("units", "")
    ans = ans.replace("{km}", "").replace("km", "")
    ans = ans.replace("step/minute", "")
    ans = ans.replace("minutes", "")
    ans = ans.replace("miles", "")
    ans = ans.replace("rs.", "")
    ans = ans.replace("rs", "")
    ans = ans.replace("days", "")
    ans = ans.replace("cm", "")
    ans = ans.replace("feet", "")
    ans = re.sub(r"\\times10\^\{(.*)\}", r"e\1", ans)
    ans = ans.replace("–", "-")
    return _strip_string(ans)


def eval_tuple(s):
    """
    (a,b,c,...)
    """
    sl = s[1:-1].split(',')
    try :
        if s[0] == '(' and s[-1] == ')' and len(sl) > 1:
            s = ','.join([str(round(eval(str(latex2sympy(sub))),2)) if 'infty' not in sub and sub not in ['a', '-a'] else sub for sub in sl])
            return f"({s})"
    except:
        return s
    return s


def is_number(s):
    try:
        complex(s)
    except:
        return False
    return True


def is_equal(asw:str, gt_asw:str) -> bool:
    """
    Judge if asw is equivalent to gt_asw.
    """
    asw = find_math_answer(asw)
    gt_asw = find_math_answer(gt_asw)
    if gt_asw == "" or asw == "":
        return False
    asw = eval_tuple(asw)
    gt_asw = eval_tuple(gt_asw)
    if gt_asw == asw:
        return True
    else:
        try:
            if ',' in gt_asw:
                if set(gt_asw.split(',')) == set(asw.split(',')):
                    return True
            if "e" in asw and "e" in gt_asw:
                asw_front = asw.split("e")[0]
                asw_end = asw.split("e")[1]
                gt_asw_front = gt_asw.split("e")[0]
                gt_asw_end = gt_asw.split("e")[1]
                if is_equal(asw_front, gt_asw_front) and is_equal(asw_end, gt_asw_end):
                    return True
            if is_number(asw) and is_number(gt_asw) and ((abs(complex(asw) - complex(gt_asw)) / (abs(complex(gt_asw)) + 1e-1000)) <= 0.1 and abs(complex(asw) - complex(gt_asw)) < 1):
                return True
            try:
                from math import sqrt
                i = 1j
                num1 = eval(str(latex2sympy(gt_asw)))
                num2 = eval(str(latex2sympy(asw)))
                try:
                    if str(round(num1, 2)) == str(round(num2, 2)):
                        return True
                except:
                    if ((abs(num1 - num2) / (abs(num2) + 1e-20)) <= 0.1 and abs(num1 - num2) < 1):
                        return True
            except:
                from math import sqrt
                i = 1j
                asw_exp = latex2sympy(asw)
                gt_asw_exp = latex2sympy(gt_asw)
                asw_exp_simplify = sp.simplify(asw_exp)
                gt_asw_exp_simplify = sp.simplify(gt_asw_exp)
                if str(asw_exp_simplify) == str(gt_asw_exp_simplify):
                    return True
                asw_exp_dict = asw_exp_simplify.as_coefficients_dict()
                gt_asw_exp_dict = gt_asw_exp_simplify.as_coefficients_dict()
                if len(asw_exp_dict.keys()) == len(gt_asw_exp_dict.keys()):
                    is_same = True
                    for key in asw_exp_dict.keys():
                        if key not in gt_asw_exp_dict or (not is_equal(str(gt_asw_exp_dict[key]), str(asw_exp_dict[key]))):
                            is_same = False
                            break
                    if is_same:
                        return True

                asw_exp_expand = sp.expand(asw_exp)
                gt_asw_exp_expand = sp.expand(gt_asw_exp)
                if str(asw_exp_expand) == str(gt_asw_exp_expand):
                    return True
                asw_exp_dict = asw_exp_expand.as_coefficients_dict()
                gt_asw_exp_dict = gt_asw_exp_expand.as_coefficients_dict()
                if len(asw_exp_dict.keys()) == len(gt_asw_exp_dict.keys()):
                    is_same = True
                    for key in asw_exp_dict.keys():
                        if key not in gt_asw_exp_dict or (not is_equal(str(gt_asw_exp_dict[key]), str(asw_exp_dict[key]))):
                            is_same = False
                            break
                    if is_same:
                        return True
            return False
        except:
            return False





def extract_last_num(text: str) -> float:
    """
    extract the last number in a string
    """
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(-?\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return None

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
    
def extract_boxed_answer(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    if len(answers) == 0:
        return ""
    else:
        return answers[-1]
    
def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def compute_accuracy(in_file, out_file, name):
    """
    compute accuracy when the answer is put in the last block and thus in completion
    """
    with open(in_file, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    correct_datas = []
    wrong_datas = []
    for data in tqdm(datas):
        gt_answer = ""
        gt_choice = ""
        if "answer" in data["extra"].keys():
            gt_answer = str(data["extra"]["answer"])
        elif "final_answer" in data["extra"].keys():
            gt_answer = str(data["extra"]["final_answer"][0])
        elif "Answer" in data["extra"].keys():
            gt_choice = data["extra"]["Answer"]
            if "A)" in data["extra"]["options"]:
                gt_answer = re.split(r"[A-Z]\) ", data["extra"]["options"])[ord(gt_choice) - ord("A") + 1]
            else:
                gt_answer = re.split(r"[A-Z]\. ", data["extra"]["options"])[ord(gt_choice) - ord("A") + 1]
        else:
            gt_choice = data["extra"]["correct"]
            gt_answer = ")".join(data["extra"]["options"][ord(gt_choice) - ord("A")].split(")")[1:])
        model_answer = extract_boxed_answer(data["model_generation"])

        data["answers"] = {"gt_answer": gt_answer, "model_answer": model_answer}
            
        if is_equal(model_answer, gt_answer) or is_equal(model_answer, gt_choice) or ('answer_val' in data['extra'].keys() and is_equal(model_answer, gt_answer)):
            correct_datas.append(data)
        else:
            wrong_datas.append(data)

    save_jsonl(correct_datas, out_file[:-6] + "_correct.jsonl")
    save_jsonl(wrong_datas, out_file[:-6] + "_wrong.jsonl")
    correct_len = len(correct_datas)
    wrong_len = len(wrong_datas)
    total_len = len(datas)
    if correct_len == 0:
        acc = 0
    else:
        acc = 100 * correct_len / total_len
    print(f"correct: {correct_len}")
    print(f"wrong: {wrong_len}")
    print(f"acc: {acc:.2f}")
    return acc, correct_len, total_len
    
def combine(in_dir):
    for name in ["GSM8K", "MATH", "SVAMP", "mathematics", "aime24", "amc23", "olympiadbench", "college_math", "carp_en", "ocw"]:
        if not os.path.exists(os.path.join(in_dir, name)):
            os.makedirs(os.path.join(in_dir, name))
        in_files = glob(os.path.join(in_dir, name, f"{name}_test_result_[0-9]*.jsonl"))
        datas = []
        for in_file in tqdm(in_files):
            datas.extend(load_jsonl(in_file))

        save_jsonl(datas, os.path.join(in_dir, name, f"{name}_test_result.jsonl"))


class Args:

    def __init__(self):
        self.ch = "3epoch"

def main():
    args = Args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "config.json"), "r") as f:
        config = json.load(f)
    
    in_dir = os.path.join("/mnt/cache/luzimu/mathllm-finetune/results/inference", config['model_name'], args.ch)
    combine(in_dir)

    accs = []
    corrects = []
    totals = []
    names =  ["GSM8K", "MATH", "SVAMP", "mathematics", "ocw", "aime24", "amc23", "carp_en", "college_math", "olympiadbench"]
    for name in names:
        dataset_in_dir = os.path.join(in_dir, name)
        print(name + ":")
        acc, correct, total = compute_accuracy(os.path.join(dataset_in_dir, f"{name}_test_result.jsonl"),
        os.path.join(dataset_in_dir, f"{name}_test_result.jsonl"), name)
        accs.append(acc)
        corrects.append(correct)
        totals.append(total)
    
    table = "|dataset|" + "|".join(names) + "|\n" + "|" + "--|" * (len(names) + 1) + "\n"
    table += "|acc|" + "|".join([f"{acc:.2f}" for acc in accs]) + "|\n"
    table += "|cor/tot|" + "|".join([f"{correct}/{total}" for correct, total in zip(corrects, totals)]) + "|\n"
    print(table)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "result.md"), "w", encoding="utf-8") as f:
        f.write(table)


if __name__ == "__main__":
    main()

    

 