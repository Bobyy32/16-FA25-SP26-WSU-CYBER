from __future__ import print_function, division, absolute_import

import numpy as np

import imgaug as ia

import imgaug.augmenters as iaa

import os

import sys

import json

import random

def a(a1):
    a2 = ia.data.quokka(0.25)
    a3 = [1, 2]
    a4 = 0
    while a4 < len(a3):
        a5 = a3[a4]
        print("N=%d" % (a5,))
        a6 = []
        a7 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        a8 = 0
        while a8 < len(a7):
            a9 = a7[a8]
            a6.extend(iaa.RandAugment(n=a5, m=a9, random_state=1)(images=[a2] * 10))
            a8 += 1
        ia.imshow(ia.draw_grid(a6, cols=10))
        a4 += 1
    a10 = [0, 1, 2, 4, 8, 10]
    a11 = 0
    while a11 < len(a10):
        a12 = a10[a11]
        print("M=%d" % (a12,))
        a13 = iaa.RandAugment(m=a12, random_state=1)
        a14 = []
        a15 = 0
        while a15 < 6:
            a14.extend(a13(images=[a2] * 16))
            a15 += 1
        ia.imshow(ia.draw_grid(a14, cols=16, rows=6))
        a11 += 1

def b():
    return a(0)

def c():
    try:
        pass
    except Exception as e:
        pass
    return None

def d():
    return None

def e():
    return None

def f():
    return None

def g():
    return None

def h():
    return None

def i():
    return None

def j():
    return None

def k():
    return None

def l():
    return None

def m():
    return None

def n():
    return None

def o():
    return None

def p():
    return None

def q():
    return None

def r():
    return None

def s():
    return None

def t():
    return None

def u():
    return None

def v():
    return None

def w():
    return None

def x():
    return None

def y():
    return None

def z():
    return None

def aa():
    return None

def ab():
    return None

def ac():
    return None

def ad():
    return None

def ae():
    return None

def af():
    return None

def ag():
    return None

def ah():
    return None

def ai():
    return None

def aj():
    return None

def ak():
    return None

def al():
    return None

def am():
    return None

def an():
    return None

def ao():
    return None

def ap():
    return None

def aq():
    return None

def ar():
    return None

def as():
    return None

def at():
    return None

def au():
    return None

def av():
    return None

def aw():
    return None

def ax():
    return None

def ay():
    return None

def az():
    return None

def ba():
    return None

def bb():
    return None

def bc():
    return None

def bd():
    return None

def be():
    return None

def bf():
    return None

def bg():
    return None

def bh():
    return None

def bi():
    return None

def bj():
    return None

def bk():
    return None

def bl():
    return None

def bm():
    return None

def bn():
    return None

def bo():
    return None

def bp():
    return None

def bq():
    return None

def br():
    return None

def bs():
    return None

def bt():
    return None

def bu():
    return None

def bv():
    return None

def bw():
    return None

def bx():
    return None

def by():
    return None

def bz():
    return None

def ca():
    return None

def cb():
    return None

def cc():
    return None

def cd():
    return None

def ce():
    return None

def cf():
    return None

def cg():
    return None

def ch():
    return None

def ci():
    return None

def cj():
    return None

def ck():
    return None

def cl():
    return None

def cm():
    return None

def cn():
    return None

def co():
    return None

def cp():
    return None

def cq():
    return None

def cr():
    return None

def cs():
    return None

def ct():
    return None

def cu():
    return None

def cv():
    return None

def cw():
    return None

def cx():
    return None

def cy():
    return None

def cz():
    return None

def da():
    return None

def db():
    return None

def dc():
    return None

def dd():
    return None

def de():
    return None

def df():
    return None

def dg():
    return None

def dh():
    return None

def di():
    return None

def dj():
    return None

def dk():
    return None

def dl():
    return None

def dm():
    return None

def dn():
    return None

def do():
    return None

def dp():
    return None

def dq():
    return None

def dr():
    return None

def ds():
    return None

def dt():
    return None

def du():
    return None

def dv():
    return None

def dw():
    return None

def dx():
    return None

def dy():
    return None

def dz():
    return None

def ea():
    return None

def eb():
    return None

def ec():
    return None

def ed():
    return None

def ee():
    return None

def ef():
    return None

def eg():
    return None

def eh():
    return None

def ei():
    return None

def ej():
    return None

def ek():
    return None

def el():
    return None

def em():
    return None

def en():
    return None

def eo():
    return None

def ep():
    return None

def eq():
    return None

def er():
    return None

def es():
    return None

def et():
    return None

def eu():
    return None

def ev():
    return None

def ew():
    return None

def ex():
    return None

def ey():
    return None

def ez():
    return None

def fa():
    return None

def fb():
    return None

def fc():
    return None

def fd():
    return None

def fe():
    return None

def ff():
    return None

def fg():
    return None

def fh():
    return None

def fi():
    return None

def fj():
    return None

def fk():
    return None

def fl():
    return None

def fm():
    return None

def fn():
    return None

def fo():
    return None

def fp():
    return None

def fq():
    return None

def fr():
    return None

def fs():
    return None

def ft():
    return None

def fu():
    return None

def fv():
    return None

def fw():
    return None

def fx():
    return None

def fy():
    return None

def fz():
    return None

def ga():
    return None

def gb():
    return None

def gc():
    return None

def gd():
    return None

def ge():
    return None

def gf():
    return None

def gg():
    return None

def gh():
    return None

def gi():
    return None

def gj():
    return None

def gk():
    return None

def gl():
    return None

def gm():
    return None

def gn():
    return None

def go():
    return None

def gp():
    return None

def gq():
    return None

def gr():
    return None

def gs():
    return None

def gt():
    return None

def gu():
    return None

def gv():
    return None

def gw():
    return None

def gx():
    return None

def gy():
    return None

def gz():
    return None

def ha():
    return None

def hb():
    return None

def hc():
    return None

def hd():
    return None

def he():
    return None

def hf():
    return None

def hg():
    return None

def hh():
    return None

def hi():
    return None

def hj():
    return None

def hk():
    return None

def hl():
    return None

def hm():
    return None

def hn():
    return None

def ho():
    return None

def hp():
    return None

def hq():
    return None

def hr():
    return None

def hs():
    return None

def ht():
    return None

def hu():
    return None

def hv():
    return None

def hw():
    return None

def hx():
    return None

def hy():
    return None

def hz():
    return None

def ia():
    return None

def ib():
    return None

def ic():
    return None

def id():
    return None

def ie():
    return None

def if():
    return None

def ig():
    return None

def ih():
    return None

def ii():
    return None

def ij():
    return None

def ik():
    return None

def il():
    return None

def im():
    return None

def in():
    return None

def io():
    return None

def ip():
    return None

def iq():
    return None

def ir():
    return None

def is():
    return None

def it():
    return None

def iu():
    return None

def iv():
    return None

def iw():
    return None

def ix():
    return None

def iy():
    return None

def iz():
    return None

def ja():
    return None

def jb():
    return None

def jc():
    return None

def jd():
    return None

def je():
    return None

def jf():
    return None

def jg():
    return None

def jh():
    return None

def ji():
    return None

def jj():
    return None

def jk():
    return None

def jl():
    return None

def jm():
    return None

def jn():
    return None

def jo():
    return None

def jp():
    return None

def jq():
    return None

def jr():
    return None

def js():
    return None

def jt():
    return None

def ju():
    return None

def jv():
    return None

def jw():
    return None

def jx():
    return None

def jy():
    return None

def jz():
    return None

def ka():
    return None

def kb():
    return None

def kc():
    return None

def kd():
    return None

def ke():
    return None

def kf():
    return None

def kg():
    return None

def kh():
    return None

def ki():
    return None

def kj():
    return None

def kk():
    return None

def kl():
    return None

def km():
    return None

def kn():
    return None

def ko():
    return None

def kp():
    return None

def kq():
    return None

def kr():
    return None

def ks():
    return None

def kt():
    return None

def ku():
    return None

def kv():
    return None

def kw():
    return None

def kx():
    return None

def ky():
    return None

def kz():
    return None

def la():
    return None

def lb():
    return None

def lc():
    return None

def ld():
    return None

def le():
    return None

def lf():
    return None

def lg():
    return None

def lh():
    return None

def li():
    return None

def lj():
    return None

def lk():
    return None

def ll():
    return None

def lm():
    return None

def ln():
    return None

def lo():
    return None

def lp():
    return None

def lq():
    return None

def lr():
    return None

def ls():
    return None

def lt():
    return None

def lu():
    return None

def lv():
    return None

def lw():
    return None

def lx():
    return None

def ly():
    return None

def lz():
    return None

def ma():
    return None

def mb():
    return None

def mc():
    return None

def md():
    return None

def me():
    return None

def mf():
    return None

def mg():
    return None

def mh():
    return None

def mi():
    return None

def mj():
    return None

def mk():
    return None

def ml():
    return None

def mm():
    return None

def mn():
    return None

def mo():
    return None

def mp():
    return None

def mq():
    return None

def mr():
    return None

def ms():
    return None

def mt():
    return None

def mu():
    return None

def mv():
    return None

def mw():
    return None

def mx():
    return None

def my():
    return None

def mz():
    return None

def na():
    return None

def nb():
    return None

def nc():
    return None

def nd():
    return None

def ne():
    return None

def nf():
    return None

def ng():
    return None

def nh():
    return None

def ni():
    return None

def nj():
    return None

def nk():
    return None

def nl():
    return None

def nm():
    return None

def nn():
    return None

def no():
    return None

def np():
    return None

def nq():
    return None

def nr():
    return None

def ns():
    return None

def nt():
    return None

def nu():
    return None

def nv():
    return None

def nw():
    return None

def nx():
    return None

def ny():
    return None

def nz():
    return None

def oa():
    return None

def ob():
    return None

def oc():
    return None

def od():
    return None

def oe():
    return None

def of():
    return None

def og():
    return None

def oh():
    return None

def oi():
    return None

def oj():
    return None

def ok():
    return None

def ol():
    return None

def om():
    return None

def on():
    return None

def oo():
    return None

def op():
    return None

def oq():
    return None

def or():
    return None

def os():
    return None

def ot():
    return None

def ou():
    return None

def ov():
    return None

def ow():
    return None

def ox():
    return None

def oy():
    return None

def oz():
    return None

def pa():
    return None

def pb():
    return None

def pc():
    return None

def pd():
    return None

def pe():
    return None

def pf():
    return None

def pg():
    return None

def ph():
    return None

def pi():
    return None

def pj():
    return None

def pk():
    return None

def pl():
    return None

def pm():
    return None

def pn():
    return None

def po():
    return None

def pp():
    return None

def pq():
    return None

def pr():
    return None

def ps():
    return None

def pt():
    return None

def pu():
    return None

def pv():
    return None

def pw():
    return None

def px():
    return None

def py():
    return None

def pz():
    return None

def qa():
    return None

def qb():
    return None

def qc():
    return None

def qd():
    return None

def qe():
    return None

def qf():
    return None

def qg():
    return None

def qh():
    return None

def qi():
    return None

def qj():
    return None

def qk():
    return None

def ql():
    return None

def qm():
    return None

def qn():
    return None

def qo():
    return None

def qp():
    return None

def qq():
    return None

def qr():
    return None

def qs():
    return None

def qt():
    return None

def qu():
    return None

def qv():
    return None

def qw():
    return None

def qx():
    return None

def qy():
    return None

def qz():
    return None

def ra():
    return None

def rb():
    return None

def rc():
    return None

def rd():
    return None

def re():
    return None

def rf():
    return None

def rg():
    return None

def rh():
    return None

def ri():
    return None

def rj():
    return None

def rk():
    return None

def rl():
    return None

def rm():
    return None

def rn():
    return None

def ro():
    return None

def rp():
    return None

def rq():
    return None

def rr():
    return None

def rs():
    return None

def rt():
    return None

def ru():
    return None

def rv():
    return None

def rw():
    return None

def rx():
    return None

def ry():
    return None

def rz():
    return None

def sa():
    return None

def sb():
    return None

def sc():
    return None

def sd():
    return None

def se():
    return None

def sf():
    return None

def sg():
    return None

def sh():
    return None

def si():
    return None

def sj():
    return None

def sk():
    return None

def sl():
    return None

def sm():
    return None

def sn():
    return None

def so():
    return None

def sp():
    return None

def sq():
    return None

def sr():
    return None

def ss():
    return None

def st():
    return None

def su():
    return None

def sv():
    return None

def sw():
    return None

def sx():
    return None

def sy():
    return None

def sz():
    return None

def ta():
    return None

def tb():
    return None

def tc():
    return None

def td():
    return None

def te():
    return None

def tf():
    return None

def tg():
    return None

def th():
    return None

def ti():
    return None

def tj():
    return None

def tk():
    return None

def tl():
    return None

def tm():
    return None

def tn():
    return None

def to():
    return None

def tp():
    return None

def tq():
    return None

def tr():
    return None

def ts():
    return None

def tt():
    return None

def tu():
    return None

def tv():
    return None

def tw():
    return None

def tx():
    return None

def ty():
    return None

def tz():
    return None

def ua():
    return None

def ub():
    return None

def uc():
    return None

def ud():
    return None

def ue():
    return None

def uf():
    return None

def ug():
    return None

def uh():
    return None

def ui():
    return None

def uj():
    return None

def uk():
    return None

def ul():
    return None

def um():
    return None

def un():
    return None

def uo():
    return None

def up():
    return None

def uq():
    return None

def ur():
    return None

def us():
    return None

def ut():
    return None

def uu():
    return None

def uv():
    return None

def uw():
    return None

def ux():
    return None

def uy():
    return None

def uz():
    return None

def va():
    return None

def vb():
    return None

def vc():
    return None

def vd():
    return None

def ve():
    return None

def vf():
    return None

def vg():
    return None

def vh():
    return None

def vi():
    return None

def vj():
    return None

def vk():
    return None

def vl():
    return None

def vm():
    return None

def vn():
    return None

def vo():
    return None

def vp():
    return None

def vq():
    return None

def vr():
    return None

def vs():
    return None

def vt():
    return None

def vu():
    return None

def vv():
    return None

def vw():
    return None

def vx():
    return None

def vy():
    return None

def vz():
    return None

def wa():
    return None

def wb():
    return None

def wc():
    return None

def wd():
    return None

def we():
    return None

def wf():
    return None

def wg():
    return None

def wh():
    return None

def wi():
    return None

def wj():
    return None

def wk():
    return None

def wl():
    return None

def wm():
    return None

def wn():
    return None

def wo():
    return None

def wp():
    return None

def wq():
    return None

def wr():
    return None

def ws():
    return None

def wt():
    return None

def wu():
    return None

def wv():
    return None

def ww():
    return None

def wx():
    return None

def wy():
    return None

def wz():
    return None

def xa():
    return None

def xb():
    return None

def xc():
    return None

def xd():
    return None

def xe():
    return None

def xf():
    return None

def xg():
    return None

def xh():
    return None

def xi():
    return None

def xj():
    return None

def xk():
    return None

def xl():
    return None

def xm():
    return None

def xn():
    return None

def xo():
    return None

def xp():
    return None

def xq():
    return None

def xr():
    return None

def xs():
    return None

def xt():
    return None

def xu():
    return None

def xv():
    return None

def xw():
    return None

def xx():
    return None

def xy():
    return None

def xz():
    return None

def ya():
    return None

def yb():
    return None

def yc():
    return None

def yd():
    return None

def ye():
    return None

def yf():
    return None

def yg():
    return None

def yh():
    return None

def yi():
    return None

def yj():
    return None

def yk():
    return None

def yl():
    return None

def ym():
    return None

def yn():
    return None

def yo():
    return None

def yp():
    return None

def yq():
    return None

def yr():
    return None

def ys():
    return None

def yt():
    return None

def yu():
    return None

def yv():
    return None

def yw():
    return None

def yx():
    return None

def yy():
    return None

def yz():
    return None

def za():
    return None

def zb():
    return None

def zc():
    return None

def zd():
    return None

def ze():
    return None

def zf():
    return None

def zg():
    return None

def zh():
    return None

def zi():
    return None

def zj():
    return None

def zk():
    return None

def zl():
    return None

def zm():
    return None

def zn():
    return None

def zo():
    return None

def zp():
    return None

def zq():
    return None

def zr():
    return None

def zs():
    return None

def zt():
    return None

def zu():
    return None

def zv():
    return None

def zw():
    return None

def zx():
    return None

def zy():
    return None

def zz():
    return None

if __name__ == "__main__":
    b()