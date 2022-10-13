import matplotlib.pyplot as plt

from mayavi.modules.surface import Surface
from mayavi.mlab import surf, gcf
from mayavi import mlab

import dionysus as dio
import numpy as np
import os, sys

class Element:
    def __init__(self, object, ctl, name):
        self._o = object
        self._o.name = name
        ctl.add_child(self._o)
        self._props = {'visible' : ['visible'],
                        'color' : ['actor', 'property', 'color'],
                        'opacity' : ['actor', 'property', 'opacity'],
                        'backface_culling' : ['actor', 'property', 'backface_culling'],
                        'lighting' : ['actor', 'property', 'lighting']}
    def _init_props(self, **kwargs):
        for k,v in kwargs.items():
            self[k] = v
    def _trait_search(self, l, set=None, p=None):
        if len(l) > 1:
            p = (self._o if p is None else p).trait_get(l[0])[l[0]]
            return self._trait_search(l[1:], set, p)
        elif set is not None:
            (self._o if p is None else p).trait_set(**{l[0] : set})
            return set
        else:
            return p.trait_get(l[0])[l[0]]
    def __getitem__(self, key):
        return self._trait_search(self._props[key])
    def __setitem__(self, key, val):
        return self._trait_search(self._props[key], val)

class SurfaceElement(Element):
    def __init__(self, ctl, name):
        Element.__init__(self, Surface(), ctl, name)
        self._o.enable_contours = True
        self._o.actor.property.lighting = False
        self._o.actor.mapper.scalar_visibility = False

class Plot:
    def __init__(self, s0):
        self.s0 = s0
        self.gcf = gcf()
        self.ctl = self.s0.parent
        self.scene = self.gcf.scene
        self.cam = self.scene.camera
    def __getitem__(self, key):
        return self._elem[key]
    def __setitem__(self, key, val):
        self._elem[key] = val
    def save(self, name, verbose=True, size=(1500*2, 868*2)):
        if verbose:
            print('saving %s' % name)
        mlab.savefig(name, size=size, magnification=1)

class SurfaceCut(SurfaceElement):
    def __init__(self, ctl, name, **kwargs):
        SurfaceElement.__init__(self, ctl, name)
        self._o.contour.filled_contours = True
        self._o.actor.property.opacity = 1
        self._props = {'min' : ['contour', 'minimum_contour'],
                        'max' : ['contour', 'maximum_contour'],
                        **self._props}
        self._init_props(**kwargs)

class SurfaceContour(SurfaceElement):
    def __init__(self, ctl, name, **kwargs):
        SurfaceElement.__init__(self, ctl, name)
        self._o.contour.filled_contours = False
        self._o.contour.auto_contours = False
        self._o.actor.property.line_width = 3
        self._o.visible = True
        self._props = {'scalar' : ['contour', 'contours'], **self._props}
        self._init_props(**kwargs)

class SurfacePlot(Plot):
    def __init__(self, X, Y, G, cuts, args, contours, view):
        Plot.__init__(self, surf(X.T, Y.T, G))
        self.s0.visible = False
        self.scene.background = (1,1,1)
        self._elem = {'cut' : {}, 'cont' : {}}
        for k, v in cuts.items():
            self['cut'][k] = SurfaceCut(self.ctl, k, **{**v, **args[k]})
        for k, v in contours.items():
            self['cont'][k] = SurfaceContour(self.ctl, k, **v)
        self._view = view
        self.reset_view(self._view['default'])
    def reset_view(self, key):
        self.set_view(**self._view[key])
    def set_view(self, view=None, zoom=None, roll=None, parallel_projection=None):
        if view is not None:
            mlab.view(*view)
        if zoom is not None:
            self.cam.parallel_scale = zoom
        if roll is not None:
            mlab.roll(roll)
        if parallel_projection is not None:
            self.scene.parallel_projection = parallel_projection
    def focus_low(self, name):
        self.focus_scalar(name)
        for k, v in self._elem['cut'].items():
            c = '%s_c' % k
            if v['max'] == self['cut'][name]['min']:
                v['opacity'] = 0.1
                v['visible'] = True
                if c in self['cont']:
                    self['cont'][c]['visible'] = True
            else:
                v['opacity'] = 0.5
                if c in self['cont']:
                    self['cont'][c]['visible'] = False
    def focus_high(self, name):
        self.focus_scalar(name)
        for k, v in self._elem['cont'].items():
            if v['scalar'][0] == self['cut'][name]['min']:
                v['visible'] = True
            else:
                v['visible'] = False
    def focus_scalar(self, name):
        for k, v in self._elem['cut'].items():
            v['opacity'] = 0.5
            if v['max'] <= self['cut'][name]['min']:
                v['visible'] = False
            else:
                v['visible'] = True
    def render(self, dir, labels, verbose=True, size=(1500*2, 868*2)):
        for l in reversed(labels):
            self.save_view(dir, 'side', l, verbose, size)
            self.save_view(dir, 'top', l, verbose, size)
            self['cut'][l]['lighting'] = False
            self['cut'][l]['opacity'] = 0.1
        self.save_view(dir, 'side', '', verbose, size)
        self.save_view(dir, 'top', '', verbose, size)
    def save_view(self, dir, key, label, verbose=True, size=(1500*2, 868*2)):
        self.reset_view(key)
        self.save(os.path.join(dir, '%s%s.png' % (key, label)), verbose, size)

class PointPlot(Plot):
    def __init__(self, P, elevation, *args, **kwargs):
        self.points, self.elevation = P, elevation
        self.Z = ((1.1 if elevation else -1.1) + np.zeros(len(P))) * elevation
        Plot.__init__(self, mlab.points3d(P[:,0], P[:,1], self.Z, *args, **kwargs))
        # b = mlab.points3d(PN[:,0], PN[:,1], Z[:], color=COLOR['red'], scale_factor=2*THRESH, opacity=0.2)
        self.s0.actor.property.lighting = False # lighting
        self.s0.actor.property.frontface_culling = False # front_culling
        self.s0.glyph.glyph_source.glyph_source.phi_resolution = 32 # res
        self.s0.glyph.glyph_source.glyph_source.theta_resolution = 32 # res

class RipsPlot(PointPlot):
    def __init__(self, P, cuts, thresh, elevation=0, **kwargs):
        # PointPlot.__init__(self, P, elevation, scale_factor=thresh, opacity=0.2) #balls
        PointPlot.__init__(self, P, elevation, scale_factor=5e-2, color=(0,0,0))
        self.rips = dio.fill_rips(P, 2, thresh)
        self.triangles = [list(s) for s in self.rips if s.dimension() == 2]
        self.s2 = mlab.triangular_mesh(P[:,0], P[:,1], self.Z, self.triangles, **kwargs)
        self.s2.visible = False
        self.s2.actor.mapper.scalar_visibility = True
        self.s2.enable_contours = True
        TRI_ARGS = {k : v for k,v in cuts.items() if v['max'] <= self.s2.contour.maximum_contour and v['min'] >= self.s2.contour.minimum_contour}
        TRI_ARGS['A'], TRI_ARGS['D']  = cuts['A'].copy(), cuts['D'].copy()
        TRI_ARGS['A']['min'] = self.s2.contour.minimum_contour
        TRI_ARGS['D']['max'] = self.s2.contour.maximum_contour
        self.s2.enable_contours = False
        self.s2.actor.mapper.scalar_visibility = False
        self.tri_cuts = {k : SurfaceCut(self.s2.parent, k, **v) for k,v in TRI_ARGS.items()}
        # for k,v in self.tri_cuts.items():
        #     v._o.actor.property.edge_visibility = True
        #     v._o.actor.property.line_width = 0.1
        #     v._o.enable_contours = False
        #
        # E = [list(s) for s in R if s.dimension()==1]
        # es = []
        # for uv in E:
        #     e = mlab.plot3d(P[uv,0], P[uv,1], Z[uv], color=COLOR['black'])#, line_width=0.01)
        #     e.parent.parent.filter.radius = 0.002
        #     e.actor.property.lighting = False
        #     e.actor.mapper.scalar_visibility = False
        #     e.parent.parent
        #     es.append(e)


    # def __contains__(self, cut):
    #     return (self.s2.contour.minimum_contour <= cut['min']
    #         and self.s2.contour.maximum_contour >= cut['max'])
    # def fix_cuts(self, cuts):
    #     a, d = cuts['A'].copy(), cuts['D'].copy()
    #     a['min'] = self.s2.contour.minimum_contour
    #     d['max'] = self.s2.contour.maximum_contour
    #     return {k : v for k,v in cuts.items() if v in self}
    #     args['A'], args['D'] = a, d


    # self.edges = [list(s) for s in R if s.dimension()==1]
    # self.s1 = []
    # for uv in E:
    #     e = mlab.plot3d(P[uv,0], P[uv,1], Z[uv], color=COLOR['black'])#, line_width=0.01)
    #     e.parent.parent.filter.radius = 0.002
    #     e.actor.property.lighting = False
    #     e.actor.mapper.scalar_visibility = False
    #     e.parent.parent
    #     self.s1.append(e)
