#!/usr/bin/env python

import sys

def readiter(fi, names=('y', 'w', 'pos', 'chk'), sep='\t'):
    seq = []
    for line in fi:
        line = line.strip('\n')
        if not line:
            yield seq
            seq = []
        else:
            fields = line.split(sep)
            if len(fields) != len(names):
                raise ValueError(
                    'Each line must have %d fields: %s\n' % (len(names), line))
            seq.append(dict(zip(names, tuple(fields))))

def apply_template(seq, t, template):
    name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
    values = []
    for field, offset in template:
        p = t + offset
        if p not in range(len(seq)):
            return None
        values.append(seq[p][field])
    return '%s=%s' % (name, '|'.join(values))

def escape(src):
    return src.replace(':', '__COLON__')

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout

    templates = []
    templates += [(('w', i),) for i in range(-2, 3)]
    templates += [(('w', i), ('w', i+1)) for i in range(-2, 2)]
    templates += [(('pos', i),) for i in range(-2, 3)]
    templates += [(('pos', i), ('pos', i+1)) for i in range(-2, 2)]
    templates += [(('chk', i),) for i in range(-2, 3)]
    templates += [(('chk', i), ('chk', i+1)) for i in range(-2, 2)]
    templates += [(('iu', i),) for i in range(-2, 3)]
    templates += [(('iu', i), ('iu', i+1)) for i in range(-2, 2)]

    for seq in readiter(fi):
        for v in seq:
            # Extract more characteristics of the input sequence
            v['iu'] = str(v['w'] and v['w'][0].isupper())

        for t in range(len(seq)):
            fo.write(seq[t]['y'])
            for template in templates:
                attr = apply_template(seq, t, template)
                if attr is not None:
                    fo.write('\t%s' % escape(attr))
            fo.write('\n')
        fo.write('\n')
