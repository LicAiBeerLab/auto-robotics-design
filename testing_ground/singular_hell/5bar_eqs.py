from matplotlib import pyplot as plt
import numpy as np
from sympy import solveset, symbols, Eq, solve, S, nsolve
from sympy.utilities.lambdify import lambdify as lambdify
from scipy.optimize import fsolve, root

import modern_robotics as mr

from auto_robot_design.description.builder import DetailedURDFCreatorFixedEE, ParametrizedBuilder
from auto_robot_design.description.mechanism import JointPoint2KinematicGraph
from auto_robot_design.description.utils import draw_links
from auto_robot_design.generator.restricted_generator.two_link_generator import TwoLinkGenerator
from testing_ground.singular_hell.workspace_run import JPGraphHandler

class KinGraphHandler:
    def __init__(self, kinematic_graph):
        self.kin_graph = kinematic_graph

        # l_order = ['L6','L5','L3','L4'] #initial
        # l_order = ['L4','L6','L3','L5']
        self.l_order = [l.name for l in (kinematic_graph.nodes()-{kinematic_graph.EE, kinematic_graph.G})]
        self.G = kinematic_graph.G.name

        j_order = []
        for j in kinematic_graph.joint_graph.nodes():
            if not kinematic_graph.EE in j.links:
                j_order.append(j.jp.name)
                # print([l.name for l in list(j.links)])
            else:
                jee = j
                ljee = list(j.links - {kinematic_graph.EE})[0]
        # nj = len(j_order)
        # nl = len(l_order)
        j_order.append(jee.jp.name)
        # print(l_order)
        # print(j_order)

        # ljee_ord = 4
        # j_order = ['Main_ground', '2L_ground','Main_knee','2L_knee','2L_bot','Main_ee']
        # #real mechanical values
        self.nl = len(kinematic_graph.nodes())-2 # -ee -G   #4
        self.nj = len(kinematic_graph.edges())-1 # -ee      #5

        self.jname2ord = {n: i for i,n in enumerate(j_order)}
        lname2ord = {n: j+1 for j,n in enumerate(self.l_order)}
        # print(lname2ord)
        lname2ord[self.G] = 0

        self.ljee_ord = lname2ord[ljee.name]

        # # print(len(kinematic_graph))
        # # print(len(main_branch))

        # # for l in main_branch:
        # #     print(l.name)

        self.links_dict = kinematic_graph.name2link
        # # jp_dict = kinematic_graph.name2jp
        # j_dict = kinematic_graph.name2joint

        # # for n in l_order:
        # #     print(len(links_dict[n].joints))

        # # for n in j_order:
        # #     print(j_dict[n].jp.name)

        # jname2lname = {}
        self.jord2lord = {}
        self.lord2jord = {}
        for j,e in kinematic_graph.joint2edge.items():
            # try:
            #     jname2lname[j.jp.name] = [l.name for l in e]
            # except(KeyError):
            #     pass
            try:
                self.jord2lord[self.jname2ord[j.jp.name]] = [lname2ord[l.name] for l in e]
                self.lord2jord[frozenset([lname2ord[l.name] for l in e])] = self.jname2ord[j.jp.name]
            except(KeyError):
                pass
            # print(j.jp.name,[l.name for l in e])

        self.w = 6 # n of vars per link
        self.norm_factor = self._calc_bounds()
        self.p_i = self._calc_joint_local_positions()

    def _calc_bounds(self, init_safety_factor: float=10.):
        # # take max distance from (0,0) to EE of 2linker as a reference length
        # if is_2linker_based:
            # ee_b = calc_ee_range_of_2linker(kinematic_graph)
        #     # norm_factor = 1./np.linalg.norm(kinematic_graph.name2joint["Main_ee"].jp.r)
        #     norm_factor = 1./ee_b[1]
            
        # else:
        #     raise NotImplementedError('The only supported base structure is two-linker, \
        #                               which is used to calculate normalization factor. \
        #                               Implement new structures or just set factor to 1.')

        norm_factor = 1.
        ee_b = np.array([-10.,10.])
        
        # Make sure that bounds contain ENTIRE range of locations for all the links' frames.
        # x_b (Union[list, tuple, np.ndarray]): bounds for 1st coord of each link's frame.
        # y_b (Union[list, tuple, np.ndarray]): bounds for 2st coord of each link's frame.
        x_b = ee_b *init_safety_factor
        y_b = ee_b *init_safety_factor

        rxb = np.asarray(x_b) *norm_factor
        ryb = np.asarray(y_b) *norm_factor

        un_b = np.array([-1., 1.])
        return norm_factor

    def _calc_joint_local_positions(self):
        norm_factor = self.norm_factor
        nj = self.nj
        nl = self.nl

        # rb_j = np.full((nl+1,2,2),None) #+g-ee
        p_i = np.full((nl+1,nj+1,2),None) #+g-ee, joints +ee, ncoords

        for ind_l, ln in enumerate([self.G,*self.l_order]):
            js = list(self.links_dict[ln].joints)
            Rot,pos = mr.TransToRp(self.links_dict[ln].frame)
            pos = pos *norm_factor
            for j in js:
                ind = self.jname2ord[j.jp.name]
                Pi = j.jp.r *norm_factor
                # print(Pi) 
                loc = Rot.T@(Pi-pos)
                loc[abs(loc)<1e-16] = 0.
                p_i[ind_l,ind,:] = loc[(0,2),]
            
        #     rb_j[ind_l,0,:] = rxb
        #     rb_j[ind_l,1,:] = ryb
        # rb_j[0,0,:] = np.zeros(2)
        # rb_j[0,1,:] = np.zeros(2)

        return p_i

    def _calc_bounds2(self):
        # B = np.zeros((nl*w+2,2))
        # for j, ln in enumerate(l_order):
        #     B[j*w,:] = rxb
        #     B[j*w+1,:] = ryb
        #     B[j*w+2,:] = un_b
        #     B[j*w+3,:] = un_b
        #     B[j*w+4,:] = pow_interval(un_b,2)
        #     B[j*w+5,:] = pow_interval(un_b,2)
        # B[(-2),:] = rxb
        # B[(-1),:] = ryb

        # #links which frames do not translate
        # gr_links = {lname2ord[l.name]: p_i[0, self.lord2jord[frozenset((0,lname2ord[l.name]))], :] 
        #             for l in kinematic_graph.neighbors(kinematic_graph.G) 
        #             if not len(p_i[lname2ord[l.name], self.lord2jord[frozenset((0,lname2ord[l.name]))], :].nonzero()[0])}

        # for j,(x,y) in gr_links.items():
        #     B[(j-1)*w,:] = (x,x)
        #     B[(j-1)*w+1,:] = (y,y)

        # print(B.T)
        pass
        

    def get_matrix_equations(self, is_sym=False,
                    is_2linker_based = True, return_lorder=False, split_selected=True
                    ):
        """
        Approximates linkage workspace with boxes. Parametrize every movable link with 6 vars: 
        (rx, ry, cos, sin, cos^2, sin^2) and the last 2 vars are x and z of EE.

        Works for mechs with only 1 joint being connected to EE. 
        
        Args:
            kinematic_graph (KinematicGraph): kinematic graph with defined EE, G fields and frames for each link.
            sigma (float): max length of a box along EE coordinates. Set it to 2 (or just below the safety factor) 
            to just shrink the initial box without subdivisions.
            init_safety_factor (float): coefficient for initial bbox that relates range of motion of any link to EE's range. 
            Needed to not miss any configurations, can be set large without any drawbacks (may crop some solutions if too low).
            is_2linker_based (bool): flag to normalize distances with max 2linker's EE distance from (0,0)

        Returns:
            list[np.ndarray]: list of solution boxes, each box contains 6*n_links bounds for intermediate 
            vars and last two bounds for two EE coordinates (x,z). n_links is len(kinematic_graph.nodes())-2 -- excluding G and EE.
        """
        w = self.w
        nj = self.nj
        nl = self.nl
        p_i = self.p_i
        # p_i = self.p_i_sym if is_sym else self.p_i
        
        nvar_1 = nl*w+2 #B.shape[0]

        A_eq = np.zeros((nj*2+nl+2,nvar_1))
        b_eq = np.zeros(nj*2+nl+2)

        for i in range(nj):
            j1,j2 = self.jord2lord[i] # 2 links' ids connected by i-th joint
            if j1 > 0: # if not ground, hence can move
                # x
                A_eq[2*i,w*(j1-1)] = 1 #rx
                A_eq[2*i,w*(j1-1)+2] = p_i[j1,i,0] # cos
                A_eq[2*i,w*(j1-1)+3] = -p_i[j1,i,1] # sin
                # y
                A_eq[2*i+1,w*(j1-1)+1] = 1 #ry
                A_eq[2*i+1,w*(j1-1)+2] = p_i[j1,i,1] # cos
                A_eq[2*i+1,w*(j1-1)+3] = p_i[j1,i,0] # sin
            else:
                b_eq[(2*i,2*i+1),] = -p_i[j1,i,:]

            if j2 > 0: # if not ground, hence can move
                # x
                A_eq[2*i,w*(j2-1)] = -1 #rx
                A_eq[2*i,w*(j2-1)+2] = -p_i[j2,i,0] # cos
                A_eq[2*i,w*(j2-1)+3] = p_i[j2,i,1] # sin
                # y
                A_eq[2*i+1,w*(j2-1)+1] = -1 #ry
                A_eq[2*i+1,w*(j2-1)+2] = -p_i[j2,i,1] # cos
                A_eq[2*i+1,w*(j2-1)+3] = -p_i[j2,i,0] # sin
            else:
                b_eq[(2*i,2*i+1),] = p_i[j2,i,:]

        sqr_pairs = []

        # quadratic equations
        for j in range(nl):
            A_eq[nj*2+j,w*j+4] = 1
            A_eq[nj*2+j,w*j+5] = 1
            b_eq[nj*2+j] = 1
            sqr_pairs.append((w*j+2,w*j+4))
            sqr_pairs.append((w*j+3,w*j+5))

        ljee_ord = self.ljee_ord

        A_eq[-2,w*(ljee_ord-1):w*(ljee_ord-1)+4] = (1,0,p_i[ljee_ord,-1,0],-p_i[ljee_ord,-1,1])
        A_eq[-2,-2] = -1
        A_eq[-1,w*(ljee_ord-1):w*(ljee_ord-1)+4] = (0,1,p_i[ljee_ord,-1,1],p_i[ljee_ord,-1,0])
        A_eq[-1,-1] = -1

        return A_eq, b_eq, sqr_pairs
    
    def get_sym_equations(self):
        return
    
    def eq_matrix2sym(self, A_eq, b_eq, sqr_pairs):
        vars = []
        vars_uniq = []
        for j in range(self.nl):
            # vars += list(symbols(f'rx_{j+1} ry_{j+1} c_{j+1} s_{j+1} c2_{j+1} s2_{j+1}'))
            rx, ry, c, s = symbols(f'rx_{j+1} ry_{j+1} c_{j+1} s_{j+1}', real=True)
            vars += [rx, ry, c, s, c**2, s**2]
            vars_uniq += [rx, ry, c, s]
        vars += list(symbols('eex eey'))
        vars_uniq += list(symbols('eex eey'))

        # print(A_eq @ vars)
        return A_eq @ vars, vars_uniq
        
    
    # def 


if __name__ == '__main__':

    gen = TwoLinkGenerator()
    # builder = ParametrizedBuilder(DetailedURDFCreatorFixedEE)
    graphs_and_cons = gen.get_standard_set()
    # np.set_printoptions(precision=3, linewidth=300, suppress=True, threshold=10000)
    graph_jp, cons_dict = graphs_and_cons[0]

    g_handler = JPGraphHandler(graph_jp)
    kin_g = g_handler.prepare_kinematic_graph(is_showed=0)


    king_handler = KinGraphHandler(kin_g)
    A_eq, b_eq, sqr_pairs = king_handler.get_matrix_equations()
    lhs, vars = king_handler.eq_matrix2sym(A_eq, b_eq, sqr_pairs)
    eqs = []
    for i, eq in enumerate(lhs):
        # print(eq.subs(1.0, 1))
        # print( Eq(eq.subs(1.0, 1), b_eq[i]) )

        eq_new = (eq-b_eq[i]).subs(1.0, 1).subs(0.0, 0)
        eqs.append(eq_new)
        # print( eq_new )

    # eqs += [S('eex')-0.5, S('eey')-0.5]
    eqs += [S('eex')-0.1, S('eey')-0.1]
    print(eqs)
    print(len(eqs), len(vars))
    print(vars)
    # sol = nsolve(eqs, vars, [1]*len(vars))
    # # sol = solveset(eqs,domain=S.Reals)
    # print(sol)

    f = lambdify([vars], eqs)
    # print(fsolve(f, [1]*len(vars)))


    def checkerboard(shape):
        return np.indices(shape).sum(axis=0) % 2
    
    divs = 4
    a = np.repeat(np.linspace(-0.5,0.5, divs), len(vars)) 
    a = a.reshape(divs,len(vars))
    # print(a)

    ar_x = a.copy()

    for i in range(divs):
        # sol = root(f, [.5]*len(vars))
        sol = root(f, a[i,:])
        print(sol.x)
        # ar_x.append(sol.x)
        ar_x[i,:] = sol.x
    # sol = root(f, [-.5]*len(vars))
    # print(sol.x)
    # sol = root(f, [0.1]*len(vars))
    # print(sol.x)
    # sol = root(f, [1.]*len(vars))
    # print(sol.x)

    # print(ar_x)
    # ar_x = np.asarray(ar_x)
    xroots = np.unique(ar_x.round(2), axis=0)
    # yroots = fd(xroots)
    print('----------------')
    print(xroots)
    



    # x, y = symbols('x y')
    # # print(solve([x+1, x>1]))
    # print(nsolve([x+1, y+2], [x,y],[1]*2))

    # nl = 2

    # p=np.array([2,7])

    # vars = []
    # for j in range(nl):
    #     vars.append(symbols(f'rx_{j+1} ry_{j+1} c_{j+1} s_{j+1} c2_{j+1} s2_{j+1}'))
    #     rx, ry, c, s, c2, s2 = vars[j]
    #     R = np.array([[c, -s],[s, c]])
    #     r = np.array([rx,ry])

    #     jnt = r + R @ p

    #     trig = s2 + c2 #== 1
    #     print(jnt[0])
    #     print(jnt[1])
    #     print(trig)
        


    # expr = x + 2*y
    # print(expr)



