"""
Library for interactive widgets

2022/23
@authors:   MS: Michael Schneider (mschneid@mbi-berlin.de)
            CK: Christopher Klose (christopher.klose@mbi-berlin.de)
"""

import numpy as np
import h5py

import scipy as sp
from scipy.ndimage import gaussian_filter
from scipy.ndimage import fourier_shift
from ipywidgets import FloatRangeSlider, FloatSlider, Button, interact, IntSlider
from scipy.constants import c, h, e

import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

import ipywidgets
import ipywidgets as widgets

import skimage.morphology

import pyFAI
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.detectors import Detector

#from fth import reconstruct, shift_image, propagate, shift_phase


def shift_image(image,shift):
    '''
    Shifts image with sub-pixel precission via Fourier space
    
    
    Parameters
    ----------
    image: array
        Moving image, will be shifted by shift vector
        
    shift: vector
        x and y translation in px
    
    Returns
    -------
    image_shifted: array
        Shifted image
    -------
    author: CK 2021
    '''
    
    #Shift Image
    shift_image = fourier_shift(sp.fft.fft2(image,workers=-1), shift)
    shift_image = sp.fft.ifft2(shift_image,workers=-1)
    shift_image = shift_image.real

    return shift_image


#Draw circle mask
def circle_mask(shape,center,radius,sigma=None):

    '''
    Draws circle mask with option to apply gaussian filter for smoothing
    
    Parameter
    =========
    shape : int tuple
        shape/dimension of output array
    center : int tuple
        center coordinates (ycenter,xcenter)
    radius : scalar
        radius of mask in px. Care: diameter is always (2*radius+1) px
    sigma : scalar
        std of gaussian filter
        
    Output
    ======
    mask: array
        binary mask, or smoothed binary mask        
    ======
    author: ck 2022
    '''
    
    #setup array
    x = np.linspace(0,shape[1]-1,shape[1])
    y = np.linspace(0,shape[0]-1,shape[0])
    X,Y = np.meshgrid(x,y)

    # define circle
    mask = np.sqrt(((X-center[1])**2+(Y-center[0])**2)) <= (radius)
    mask = mask.astype(float)

    # smooth aperture
    if np.logical_or(sigma != None,sigma != 0):
        mask = gaussian_filter(mask,sigma)
           
    return mask

def cimshow(im, **kwargs):
    """Simple 2d image plot with adjustable contrast.
    
    Returns matplotlib figure and axis created.
    """
    im = np.array(im).astype("float")
    fig, ax = plt.subplots(figsize=(6,6))
    im0 = im[0] if len(im.shape) == 3 else im
    mm = ax.imshow(im0, **kwargs)

    cmin, cmax, vmin, vmax = np.nanpercentile(im, [.1, 99.9, .001, 99.999])
    # vmin, vmax = np.nanmin(im), np.nanmax(im)
    sl_contrast = FloatRangeSlider(
        value=(cmin, cmax), min=vmin, max=vmax, step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width='500px'),
    )

    @ipywidgets.interact(contrast=sl_contrast)
    def update(contrast):
        mm.set_clim(contrast)
    
    if len(im.shape) == 3:
        w_image = IntSlider(value=0, min=0, max=im.shape[0] - 1)
        @ipywidgets.interact(nr=w_image)
        def set_image(nr):
            mm.set_data(im[nr])
    
    
    return fig, ax


def adjust_contrast(axis):
    """Change contrast of image displayed in axis."""
    img = axis.images[-1]
    vmin, vmax = img.get_clim()
    sl_contrast = ipywidgets.FloatRangeSlider(
        value=(vmin, vmax),
        min=img.get_array().min(),
        max=img.get_array().max(),
        step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width="500px"),
    )

    @ipywidgets.interact(contrast=sl_contrast)
    def update(contrast):
        img.set_clim(contrast)


class InteractiveCenter:
    """Plot image with controls for contrast and beamstop alignment tools."""
    
    def __init__(self, im, c0=None, c1=None, rBS=15, **kwargs):
        im = np.array(im)
        self.fig, self.ax = cimshow(im, **kwargs)
        self.mm = self.ax.get_images()[0]
        
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        
        self.c0 = c0
        self.c1 = c1
        self.rBS = rBS
        
        self.circles = []
        for i in range(5):
            color = 'g' if i == 1 else 'r'
            circle = plt.Circle([c0, c1], 10 * (i + 1), ec=color, fill=False)
            self.circles.append(circle)
            self.ax.add_artist(circle)

        self.widgets = { "w_c0" :ipywidgets.IntText(value=c0,step = 0.5, description="c0 (vert)"),
        "w_c1" : ipywidgets.IntText(value=c1,step = 0.5, description="c1 (hor)"),
        "w_rBS" : ipywidgets.IntText(value=rBS, description="rBS")}
        
        ipywidgets.interact(self.update, c0=self.widgets["w_c0"], c1=self.widgets["w_c1"], r=self.widgets["w_rBS"])
        self.fig.canvas.mpl_connect("button_press_event", self.onclick_handler)
    
    def update(self, c0, c1, r):
        self.c0 = c0
        self.c1 = c1
        self.rBS = r
        for i, c in enumerate(self.circles):
            c.set_center([c1, c0])
            c.set_radius(r * (i + 1))
            
    def onclick_handler(self, event):
        """Set the center of the active circle to clicked position."""
        if event.button == 3:  # MouseButton.RIGHT:
            c1, c0 = (event.xdata, event.ydata)
            self.widgets["w_c0"].value = int(c0)
            self.widgets["w_c1"].value = int(c1)
            self.update(int(c0),int(c1),self.rBS)

def axis_to_roi(axis, labels=None):
    """
    Generate numpy slice expression from bounds of matplotlib figure axis.
    
    If labels is not None, return a roi dictionary for xarray.
    """
    x0, x1 = sorted(axis.get_xlim())
    y0, y1 = sorted(axis.get_ylim())
    if labels is None:
        roi = np.s_[
            int(round(y0)):int(round(y1)),
            int(round(x0)):int(round(x1))
        ]
    else:
        roi = {
            labels[0]: slice(int(round(y0)), int(round(y1))),
            labels[1]: slice(int(round(x0)), int(round(x1)))
        }
    return roi


def intensity_scale(im1, im2, mask=None):
    mask = mask if mask is not None else 1
    diff = (im1 - im2) * mask
    fig, ax = plt.subplots()
    hist, bins, patches = ax.hist(mask.flatten(), np.linspace(-100, 100, 201))
    ax.set_yscale("log")
    ax.axvline(0, c='r', lw=.5)
    ax.grid(True)

    @ipywidgets.interact(f=(.2, 2.0, .001))
    def update(f):
        diff = mask * (im1 - f * im2)
        hist, _ = np.histogram(diff, bins)
        for p, v in zip(patches, hist):
            p.set_height(v)
    return fig, ax
    
    
    
class AzimuthalIntegrationCenter:
    """Plot image with controls for contrast and center alignment tools."""

    def __init__(self, im, ai, c0=None, c1=None, mask=None,circle_radius=100,**kwargs):
        # User Feedback/Instructions
        print("Left: 1d azimuthal Integration I(q)")
        print("Right: 2d azimuthal Integration I(q,chi)")
        print("Use arrow buttons on keyboard to adjust center position after selecting a slider.") 
        print("Try to transform all rings of the Airy pattern into a straight line in the 2d I(q,chi)-plot. Maximize fringe contrast in 1d I(q) plot for fine-tuning.")
        
        # Get center
        self.im = np.array(im)
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        
        #Variables
        self.c0 = c0
        self.c1 = c1
        self.radial_range = kwargs["radial_range"]
        self.im_data_range = kwargs["im_data_range"]
        self.pixel_size1 = ai.detector.get_pixel1()
        self.pixel_size2 = ai.detector.get_pixel2()
        self.qlines = kwargs["qlines"]
        self.ai = ai
        self.mask = mask

        # Calc azimuthal integration
        self.I_t, self.q_t, self.phi_t = self.ai.integrate2d(
            self.im,
            500,
            radial_range=self.radial_range,
            unit="q_nm^-1",
            correctSolidAngle=False,
            dummy=np.nan,
            mask = self.mask,
            method = "BBox"
        )
        self.mI_t = np.nanmean(self.I_t, axis=0)

        # Plot
        self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 4))    
        # center widget
        mi, ma = np.nanpercentile(self.I_t, self.im_data_range)
        self.ax[0].imshow(im, vmin=mi, vmax=ma)
        self.circles = []        
        
        for i in range((im.shape[0]//2//circle_radius)):
            color = 'g' if i == 1 else 'r'
            circle = plt.Circle([c0, c1], circle_radius * (i + 1), ec=color, fill=False, alpha=0.5)
            self.circles.append(circle)
            self.ax[0].add_artist(circle)
            
        # 1d Ai
        self.ax[1].plot(self.q_t, self.mI_t)
        self.ax[1].set_xlim(self.radial_range)
        self.ax[1].set_xlabel("q in 1/nm")
        self.ax[1].set_ylabel("Mean Integrated Intensity")
        self.ax[1].grid()
        
        # 2d Ai
        self.timshow = self.ax[2].imshow(self.I_t, vmin=mi, vmax=ma)
        self.ax[2].set_ylabel("Angle")
        self.ax[2].set_xlabel("q in px")
        self.ax[2].grid()

        # qlines
        for qt in self.qlines:
            self.ax[2].axvline(qt, ymin=0, ymax=360, c="red")

        w_c0 = ipywidgets.FloatSlider(value=c0,min=im.shape[-2]/2-np.round(im.shape[-2]/2),max=im.shape[-2]/2+np.round(im.shape[-2]/2),step=.5, description="y-center",layout=ipywidgets.Layout(width="500px"))
        w_c1 = ipywidgets.FloatSlider(value=c1,min=im.shape[-1]/2-np.round(im.shape[-1]/2),max=im.shape[-1]/2+np.round(im.shape[-1]/2),step=.5, description="x-center",layout=ipywidgets.Layout(width="500px"))

        ipywidgets.interact(self.update, c0=w_c0, c1=w_c1)

    def update(self, c0, c1, **kwargs):
        self.c0 = c0
        self.c1 = c1

        self.ai.poni1 = (
            self.c0 * self.pixel_size1)  # y (vertical)
        self.ai.poni2 = (
            self.c1 * self.pixel_size2)  # x (horizontal)

        self.I_t, self.q_t, self.phi_t = self.ai.integrate2d(
            self.im,
            500,
            radial_range=self.radial_range,
            unit="q_nm^-1",
            correctSolidAngle=False,
            dummy=np.nan,
            mask = self.mask,
            method = "BBox"
        )
        self.mI_t = np.nanmean(self.I_t, axis=0)

        # Plot
        #plot center
        for i, c in enumerate(self.circles):
            c.set_center([c1, c0])
 
        
        # 1d Ai
        self.ax[1].clear()
        self.ax[1].plot(self.q_t, self.mI_t)
        self.ax[1].set_xlabel("q in 1/nm")
        self.ax[1].set_ylabel("Mean Integrated Intensity")
        self.ax[1].grid()

        # 2d Ai
        mi, ma = np.nanpercentile(self.I_t, self.im_data_range)
        self.timshow.set_data(self.I_t)
        self.timshow.set_clim([mi, ma])
        
        
class InteractiveCircleCoordinates:
    """For drawing of circles, i.e., phase retrieval mask."""

    def __init__(self, im, nr_circ, coordinates="None", **kwargs):
        # Display image
        im = np.array(im)
        self.fig, self.ax = cimshow(im, **kwargs)
        self.mm = self.ax.get_images()[0]

        # Create list of aperture coordinates
        if coordinates == "None":
            self.c_yxr = []
            for i in range(nr_circ):
                self.c_yxr.append([im.shape[-2] // 2, im.shape[-1] // 2, 15])
        else:
            self.c_yxr = coordinates

        # Draw circles
        self.circles = []
        for i in range(nr_circ):
            color = "r" if i == 0 else "g"
            circle = plt.Circle(
                [self.c_yxr[i][1], self.c_yxr[i][0]],
                self.c_yxr[i][2],
                ec=color,
                fill=False,
            )
            self.circles.append(circle)
            self.ax.add_artist(circle)

        # Create slider to select aperture
        w_idx = ipywidgets.IntSlider(
            value=0,
            min=0,
            max=nr_circ - 1,
            step=1,
            description="circle index",
            layout=ipywidgets.Layout(width="250px"),
        )

        # Create circle widget sliders
        label = ["yc", "xc", "r0"]
        width = ["600px", "600px", "500px"]
        mi = [0, 0, 0]
        ma = [2*im.shape[-2], 2*im.shape[-1], 2*np.max([im.shape[-2],im.shape[-1]])]
        w = [
            ipywidgets.FloatSlider(
                value=self.c_yxr[0][k],
                min=mi[k],
                max=ma[k],
                step=0.5,
                description=label[k],
                layout=ipywidgets.Layout(width=width[k]),
            )
            for k in range(3)
        ]
        self.w = w

        # Interactive user interface
        iidx = ipywidgets.interact(self.update_index, idx_circ=w_idx)
        icirc = ipywidgets.interact(
            self.update,
            c0=self.w[0],
            c1=self.w[1],
            r=self.w[2],
        )

    # Update initial widget values to the ones from list when changing between circles
    def update_index(self, idx_circ):
        self.idx = idx_circ
        cy = self.c_yxr[self.idx][0]
        cx = self.c_yxr[self.idx][1]
        cr = self.c_yxr[self.idx][2]

        # Keep these separated as this prevents overwriting of the individual widget values
        self.w[0].value = cy
        self.w[1].value = cx
        self.w[2].value = cr
        
        #Change color, active circle is red
        for i, c in enumerate(self.circles):
            color = "r" if i == self.idx else "g"
            c.set_edgecolor(color)

    # Update circle values
    def update(self, c0, c1, r):
        # Update coordinate dictionary
        self.c_yxr[self.idx] = [c0, c1, r]

        # Update drawn circles
        c = self.circles[self.idx]
        c.set_center([c1, c0])
        c.set_radius(r)

        print("Aperture Coordinates:")
        print(self.c_yxr)
        
        
class InteractiveBeamstop:
    """Plot image with controls for contrast and beamstop alignment tools."""
    def __init__(self, im, c0=None, c1=None, rBS=60,stdBS=4, **kwargs):        
        #Parameter coordinates
        if c0 is None:
            c0 = im.shape[-2] // 2
        if c1 is None:
            c1 = im.shape[-1] // 2
        self.center = [c0,c1]
        
        #Beamstop parameter
        self.rBS = rBS
        self.stdBS = stdBS
        
        # Create beamstop mask
        im = np.array(im)
        self.im = im
        self.mask_bs = 1 - circle_mask(
            im.shape, self.center, self.rBS, sigma = self.stdBS
        )
        self.image = np.array(im*self.mask_bs)
        
        #Plotting
        fig, ax = plt.subplots()
        self.mm = ax.imshow(self.image)
        cmin, cmax, vmin, vmax = np.nanpercentile(im, [.1, 99, .1, 99.9])
        sl_contrast = FloatRangeSlider(
        value=(cmin, cmax), min=vmin, max=vmax, step=(vmax - vmin) / 500,
        layout=ipywidgets.Layout(width='500px'),
        )
        cim = ipywidgets.interact(self.update_plt, contrast = sl_contrast)
        
        #Change beamstop parameter
        w_rBS = ipywidgets.IntText(value=self.rBS, description="rBS")
        w_std = ipywidgets.IntText(value=self.stdBS, description="stdBS")
        ipywidgets.interact(self.update_bs, r=w_rBS,std = w_std)
    
    #Update plot
    def update_plt(self,contrast):
        self.mm.set_clim(contrast)
    
    #Update bs
    def update_bs(self, r,std):
        self.rBS = r
        self.stdBS = std
        self.mask_bs = 1 - circle_mask(
            self.mask_bs.shape, self.center, r, sigma = std
        )
        self.image = self.im*self.mask_bs
        self.mm.set_data(self.image)

class draw_polygon_mask:
    """Interactive drawing of polygon masks"""

    def __init__(self, image,**kwargs):
        self.image = image
        self.image_plot = image
        self.full_mask = np.zeros(image.shape)
        self.coordinates = []
        self.masks = []
        self._create_widgets()
        self.kwargs = kwargs
        self.draw_gui()

    def _create_widgets(self):
        self.button_add = ipywidgets.Button(
            description="Add mask",
            button_style="warning",
            layout=ipywidgets.Layout(height="auto", width="100px"),
        )
        self.button_add.on_click(self.add_mask)
        
        
        self.button_del = ipywidgets.Button(
            description="Delete mask",
            #button_style="warning",
            layout=ipywidgets.Layout(height="auto", width="100px"),
        )
        self.button_del.on_click(self.del_mask)

    def draw_gui(self):
        """Create plot and control widgets"""

        # Plotting
        fig, self.ax = plt.subplots(figsize= (8,8))
        self.mm = self.ax.imshow(self.image_plot,**self.kwargs)
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.01, 99.99, 0.01, 99.99])

        sl_contrast = FloatRangeSlider(
            value=(cmin, cmax),
            min=vmin,
            max=vmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
        )
        cim = ipywidgets.interact(self.update_plt, contrast=sl_contrast)

        # How to use
        print("Click on the figure to create a polygon corner.")
        print("Click `Add mask` to store coordinates and apply mask.")
        print("Press the 'esc' key to reset the polygon for new drawing.")
        print("")
        print("Try holding the 'shift' key to move all of the vertices.")
        print("Try holding the 'ctrl' key to move a single vertex.")
        print("Button `Delete mask` deletes the masks recursively.")
        

        self.reset_polygon_selector()
        self.output = ipywidgets.Output()
        display(self.button_add,self.button_del, self.output)

    # Update plot
    def update_plt(self, contrast):
        self.mm.set_clim(contrast)

    def reset_polygon_selector(self):
        self.selector = PolygonSelector(
            self.ax,
            lambda *args: None,
            props=dict(color="r", linestyle="-", linewidth=2, alpha=0.9),
        )

    def create_polygon_mask(self, shape, coordinates):
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x, y)).T

        path = Path(coordinates)
        mask = path.contains_points(points)
        mask = mask.reshape(shape)
        self.masks.append(mask)
        self.coordinates.append(coordinates)
        
    def combine_masks(self):
        if len(self.masks) == 0:
            self.full_mask = np.zeros(self.image.shape)
        if len(self.masks) == 1:
            self.full_mask = self.masks[0]
        elif len(self.masks) > 1:
            self.full_mask = np.sum(np.array(self.masks).astype(int), axis=0)

        self.full_mask[self.full_mask > 1] = 1

    def add_mask(self, change):
        self.create_polygon_mask(self.image.shape, self.selector.verts)
        self.combine_masks()
        self.image_plot = self.image * (1 - self.full_mask)
        self.mm.set_data(self.image_plot)
        
    def del_mask(self,change):
        self.coordinates.pop()
        self.masks.pop()
        self.combine_masks()
        self.image_plot = self.image * (1 - self.full_mask)
        self.mm.set_data(self.image_plot)
        
    def round_nested_list(self, nested_list, precision):
        """
        Round all values in a nested list to a specified precision.

        Args:
        nested_list (list): The nested list containing numerical values and/or tuples.
        precision (int): Number of decimal places to round to (default is 2).

        Returns:
        list: A new nested list with all values rounded to the specified precision.
        """
        
        if isinstance(nested_list, list):
            return [self.round_nested_list(item, precision) for item in nested_list]
        elif isinstance(nested_list, tuple):
            return tuple(round(value, precision) for value in nested_list)
        else:
            return round(nested_list, precision)
        
    def get_vertice_coordinates(self):
        return self.round_nested_list(self.coordinates,1)


class Shift_Scale_Mask:
    """Plot image with controls for contrast, x/y shift and scaling."""
    
    def __init__(self, image, mask, shift = [0,0], scale = 0, **kwargs):
        self.image = image
        self.shape = self.image.shape
        self.mask_original = mask
        self.mask = mask
        self.mask_shifted = mask
        self.shift = shift
        self.scale = scale
        self.kwargs = kwargs
        self.draw_gui()

        
    def draw_gui(self):
        """Create plot and control widgets."""

        self.fig, self.ax = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.01, 99.99, 0.1, 99.9])
        self.m0 = self.ax[0].imshow(self.image,**self.kwargs)
        self.m1 = self.ax[1].imshow(self.image,**self.kwargs)
        self.ax[0].set_title("Image*Mask")
        self.ax[1].set_title("Image*(1-Mask)")
            
        self.widgets = {
            "contrast": widgets.FloatRangeSlider(
            value=(vmin, vmax),
            min=cmin,
            max=cmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
            ),
            "shift_ver": widgets.FloatSlider(
                min=-self.shape[1]/4, max=self.shape[1]/4, value=self.shift[0], step=0.5, description="shift_ver",layout=ipywidgets.Layout(width="350px")),
            "shift_hor": widgets.FloatSlider(
                min=-self.shape[1]/4, max=self.shape[1]/4, value=self.shift[1], step=0.5, description="shift_hor",layout=ipywidgets.Layout(width="350px")),
            "scale": widgets.IntSlider(min=-20, max=20, value=self.scale,description="scale"),
                    }

        ipywidgets.interact(self.update_plt_contrast, contrast=self.widgets["contrast"])
        widgets.interact(
            self.update_mask,
            shift_ver=self.widgets["shift_ver"],
            shift_hor=self.widgets["shift_hor"],
            scale=self.widgets["scale"],
        )
        
        self.fig.canvas.mpl_connect("button_press_event", self.onclick_handler)
            
    def update_plt_contrast(self, contrast):
        self.m0.set_clim(contrast)
        self.m1.set_clim(contrast)
    
    def update_plt_images(self):
        self.m0.set_data(self.image*self.mask)
        self.m1.set_data(self.image*(1-self.mask))
    
    def shift_mask(self, shift_ver,shift_hor):
            self.shift = [shift_ver,shift_hor]
            self.mask = np.round(shift_image(self.mask_original,self.shift))
        
    def scale_mask(self,scale):
        self.scale = -scale
        if scale > 0:
            footprint = skimage.morphology.disk(scale)
            self.mask = skimage.morphology.dilation(self.mask, footprint)
        elif scale < 0:
            footprint = skimage.morphology.disk(np.abs(scale))
            self.mask = skimage.morphology.erosion(self.mask, footprint)
            
    def update_mask(self,shift_ver,shift_hor,scale):
        self.shift_mask(shift_ver,shift_hor)
        if scale !=0:
            self.scale_mask(scale)
            
        self.update_plt_images()
            
    def onclick_handler(self, event):
        """Set the center of the mask to clicked position."""
        if event.button == 3:  # MouseButton.RIGHT:
            c0, c1 = (event.xdata, event.ydata)
            shift = [self.mask.shape[0]/2-c0,self.mask.shape[0]/2-c1]
            self.update_mask(shift[0],shift[1],self.scale)
        
    def get_mask(self):
        """Return list of tuples with mask parameters (center, radius)"""
        return self.mask, self.shift, self.scale

    
class Shift_Rotate:
    """Plot image with controls for contrast, x/y shift and scaling."""
    
    def __init__(self, image, shift = [0,0], angle = 0, ticks = None):
        self.image = image
        self.shape = self.image.shape
        self.image_original = image
        self.shift = shift
        self.angle = angle
        self.ticks = ticks
        
        self.draw_gui()        
        
    def draw_gui(self):
        """Create plot and control widgets."""

        self.fig, self.ax = plt.subplots(figsize=(8,8))
        cmin, cmax, vmin, vmax = np.nanpercentile(self.image, [0.01, 99.99, 0.01, 99.99])
        self.m0 = self.ax.imshow(self.image,vmin=vmin,vmax=vmax)
        self.ax.set_title("Image")
            
        if self.ticks is not None:
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            self.ax.set_xticks(self.ticks[1])
            self.ax.set_yticks(self.ticks[0])
            plt.grid()
            
            
        self.widgets = {
            "contrast": widgets.FloatRangeSlider(
            value=(vmin, vmax),
            min=cmin,
            max=cmax,
            step=(vmax - vmin) / 500,
            layout=ipywidgets.Layout(width="500px"),
            ),
            "shift_ver": widgets.FloatSlider(
                min=-self.shape[1]/2, max=self.shape[1]/2, value=self.shift[0], step=0.5, description="shift_ver",layout=ipywidgets.Layout(width="350px")),
            "shift_hor": widgets.FloatSlider(
                min=-self.shape[1]/2, max=self.shape[1]/2, value=self.shift[1], step=0.5, description="shift_hor",layout=ipywidgets.Layout(width="350px")),
            "angle": widgets.FloatSlider(min=-180, max=180, value=self.angle,step=0.25,description="angle"),
                    }

        ipywidgets.interact(self.update_plt_contrast, contrast=self.widgets["contrast"])
        widgets.interact(
            self.update_image,
            shift_ver=self.widgets["shift_ver"],
            shift_hor=self.widgets["shift_hor"],
            angle=self.widgets["angle"],
        )
        
        self.fig.canvas.mpl_connect("button_press_event", self.onclick_handler)
            
    def update_plt_contrast(self, contrast):
        self.m0.set_clim(contrast)
    
    def update_plt_images(self):
        self.m0.set_data(self.image)
        
    def update_image(self,shift_ver,shift_hor,angle):
        self.rotate_image(angle)
        self.shift_image(shift_ver,shift_hor)
            
        self.update_plt_images()
    
    def update_controls(self,shift):
        self.widgets["shift_ver"].value = shift[0]
        self.widgets["shift_hor"].value = shift[1]
    
    def shift_image(self, shift_ver,shift_hor):
            self.shift = [shift_ver,shift_hor]
            self.image = np.round(shift_image(self.image,self.shift))
        
    def rotate_image(self,angle):
        self.angle = angle
        
        if self.angle != 0:
            self.image = rotate(self.image_original,self.angle,reshape=False)
        elif self.angle == 0:
            self.image = self.image_original.copy()
            
    def onclick_handler(self, event):
        """Set the center of the active circle to clicked position."""
        if event.button == 3:  # MouseButton.RIGHT:
            c0, c1 = (event.xdata, event.ydata)
            shift = [-1*(self.shape[0]/2-c1),-1*(self.shape[1]/2-c0)]
            self.update_controls(shift)
            self.update_image(shift[0],shift[1],self.angle)
        
    def get_parameter(self):
        """Return list of tuples with mask parameters (center, radius)"""
        return self.image, self.shift, self.angle