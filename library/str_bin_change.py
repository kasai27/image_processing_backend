import textwrap

def str_to_bin(s):
    s_bin = ''.join(format(ord(x), '08b') for x in s)
    return s_bin.ljust(1024,'0')

def bin_to_str(s_bin):
    n = 2
    s_list = []
    s_bin_list = textwrap.wrap(s_bin, width=8)
    for i in range(len(s_bin_list)):
        s_bin_10 = 0
        s_bin_len = 7
        s_bin = s_bin_list[i]
        for s in str(s_bin):
            tmp = int(s) * (n ** s_bin_len)
            s_bin_10 += tmp
            s_bin_len -= 1
        s_list.append(chr(s_bin_10))
        s = ''.join(s_list)
    return s

if __name__ == "__main__":
    s_bin = str_to_bin("aabc bff")
    print(s_bin)
    s = bin_to_str(s_bin)
    print(s)