import sys
import getopt

from engine import *

def main(argv):
    if os.path.exists('result') == False:
        os.mkdir('result')
        os.mkdir('result/image')
        os.mkdir('result/char')

    img_path = ''
    print_img = False
    try:
        opts, _ = getopt.getopt(argv, 'hi:o', ['input=', 'print_image'])

        for opt, arg in opts:
            if opt == '--input':
                img_path = arg
            if opt == '--print_image':
                print_img = True

        if input == '':
            print('Please specify input path')
            sys.exit(2)

    except getopt.GetoptError:
        print('Command line arguments invalid, try again')
        sys.exit(2)

    OCNE = OCR_Chu_Nom_Engine()

    OCNE.pipeline(image_path = img_path, print_img = print_img)

if __name__ == '__main__':
    main(sys.argv[1:])