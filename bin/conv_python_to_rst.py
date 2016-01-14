# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:19:07 2016

@author: edouard.duchesnay@cea.fr
"""

import sys, os, argparse

txt_prefix = '## '

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input pyhton file')

    options = parser.parse_args()

    if not options.input:
        print >> sys.stderr, 'Required input file'
        sys.exit(os.EX_USAGE)
    input_filename = options.input
    #input_filename = "/home/ed203246/git/pylearn-doc/src/tools_numpy.py"
    output_filename = os.path.splitext(input_filename)[0] + ".rst"
    input_fd = open(input_filename, 'r')
    output_fd = open(output_filename, 'w')
    
    #line_in = '## Pandas data manipulation'
    new_block = False
    for line_in in input_fd:
        print(line_in, len(line_in))
        if len(line_in.strip()) == 0:
            output_fd.write(line_in)
        elif line_in[:len(txt_prefix)] == txt_prefix:
            output_fd.write(line_in[len(txt_prefix):])
            new_block = True
        else:
            if new_block:
                output_fd.write('.. code:: python\n\n')
                new_block = False
                output_fd.write('    ' + line_in)
            else:
                output_fd.write('    ' + line_in)
    #output_fd.write("toto")
    input_fd.close()
    output_fd.close()
