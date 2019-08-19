"""
This program converts the text annotations to .arff format (compatible with Weka)
It also checks to ensure that all of the annotations are present in the script.
"""


def open_file(filename):
    with open(filename, "r"):
        lines = [p.split() for p in [line.rstrip('\n') for line in open(filename)]]

    return lines


lines = open_file('data/experiment_scripts/ex6_script.txt')
lines_to_fix = []
i = 0

for line in lines:
    arff_line = []
    i += 1

    if len(line) > 7:  # the min len of the lines we want to consider
        norm = line[6][1]  # D/B/P
        ia = line[1][0]  # IA: Y or N
        pfh = line[3][0]  # PFH: Y or N
        tp = line[5][0]  # TP: Y or N

        if pfh == 'Y':
            pfh_str = 'PFH: Y'
            arff_line.append(pfh_str)
        elif pfh == 'N':
            pfh_str = 'PFH: N'
            arff_line.append(pfh_str)
        else:
            lines_to_fix.append(i)

        if ia == 'Y':
            ia_str = 'IA: Y'
            arff_line.append(ia_str)
        elif ia == 'N':
            ia_str = 'IA: N'
            arff_line.append(ia_str)
        else:
            lines_to_fix.append(i)

        if tp == 'Y':
            tp_str = 'TP: Y'
            arff_line.append(tp_str)
        elif tp == 'N':
            tp_str = 'TP: N'
            arff_line.append(tp_str)
        else:
            lines_to_fix.append(i)

        if norm == 'D':
            d_str = 'D: Y'
            arff_line.append(d_str)
            b_str = 'B: N'
            arff_line.append(b_str)
            p_str = 'P: N'
            arff_line.append(p_str)
        elif norm == 'B':
            d_str = 'D: N'
            arff_line.append(d_str)
            b_str = 'B: Y'
            arff_line.append(b_str)
            p_str = 'P: Y'
            arff_line.append(p_str)
        elif norm == 'P':
            d_str = 'D: N'
            arff_line.append(d_str)
            b_str = 'B: N'
            arff_line.append(b_str)
            p_str = 'P: Y'
            arff_line.append(p_str)
        else:
            lines_to_fix.append(i)

        new_line = ','.join(arff_line)
        print(new_line)

if len(lines_to_fix) > 0:
    print(lines_to_fix)
