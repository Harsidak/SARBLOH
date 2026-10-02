"""Screenshot -> 64x64 palette-index grid using the official ARC-AGI-3 16-colour palette."""
import numpy as np, sys
from PIL import Image
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baseline'))
from arc_eye import fit_lattice
PAL = {0:'#FFFFFF',1:'#CCCCCC',2:'#999999',3:'#666666',4:'#333333',5:'#000000',6:'#E53AA3',7:'#FF7BCC',
       8:'#F93C31',9:'#1E93FF',10:'#88D8F1',11:'#FFDC00',12:'#FF851B',13:'#921231',14:'#4FCC30',15:'#A356D6'}
RGB = np.array([[int(h[i:i+2],16) for i in (1,3,5)] for h in PAL.values()], float)
def shot_to_grid(path):
    im = np.asarray(Image.open(path).convert('RGB')).astype(float)
    H, W, _ = im.shape
    lat = fit_lattice(im); P = lat.pitch
    out = np.full((lat.rows, lat.cols), -1, int); m = max(1, int(round(P*0.25)))
    for r in range(lat.rows):
        for c in range(lat.cols):
            ya, yb = max(0,int(round(lat.y0+r*P))+m), min(H,int(round(lat.y0+(r+1)*P))-m)
            xa, xb = max(0,int(round(lat.x0+c*P))+m), min(W,int(round(lat.x0+(c+1)*P))-m)
            if yb-ya < 1 or xb-xa < 1: continue
            px = np.median(im[ya:yb, xa:xb].reshape(-1,3), 0)
            out[r, c] = int(np.argmin(((RGB-px)**2).sum(1)))
    return out, lat
if __name__ == '__main__':
    g, lat = shot_to_grid(sys.argv[1]); print(g.shape, lat)
    print('\n'.join(''.join('0123456789abcdef'[v] if v>=0 else '?' for v in row) for row in g))
