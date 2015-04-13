#!/usr/bin/env python

from __future__ import print_function

import math
import logging
import argparse
import numpy as np
from PIL import Image
import geoip2.database
from PIL import ImageColor
from itertools import imap
from colorsys import hsv_to_rgb
from collections import defaultdict


__version__ = '0.0.2'


class LinearKernel:
    '''Uses a linear falloff, essentially turning a point into a cone.'''
    def __init__(self, radius):
        self.radius = radius  # in pixels
        self.radius_float = float(radius)  # worthwhile time saver

    def heat(self, distance):
        if distance >= self.radius:
            return 0.0
        return 1.0 - (distance / self.radius_float)


class GaussianKernel:
    def __init__(self, radius):
        '''radius is the distance beyond which you should not bother.'''
        self.radius = radius
        # We set the scale such that the heat value drops to 1/256 of
        # the peak at a distance of radius.
        self.scale = math.log(256) / radius

    def heat(self, distance):
        '''Returns 1.0 at center, 1/e at radius pixels from center.'''
        return math.e ** (-distance * self.scale)


class Coordinate(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    first = property(lambda self: self.x)
    second = property(lambda self: self.y)

    def copy(self):
        return self.__class__(self.first, self.second)

    def __str__(self):
        return '(%s, %s)' % (str(self.x), str(self.y))

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, o):
        return True if self.x == o.x and self.y == o.y else False

    def __sub__(self, o):
        return self.__class__(self.first - o.first, self.second - o.second)


class Projection(object):
    # For guessing scale, we pretend the earth is a sphere with this
    # radius in meters, as in Web Mercator (the projection all the
    # online maps use).
    EARTH_RADIUS = 6378137  # in meters

    def get_pixels_per_degree(self):
        try:
            return self._pixels_per_degree
        except AttributeError:
            raise AttributeError('projection scale was never set')

    def set_pixels_per_degree(self, val):
        self._pixels_per_degree = val
        logging.info('scale: %f meters/pixel (%f pixels/degree)'
                     % (self.meters_per_pixel, val))

    def get_meters_per_pixel(self):
        return 2 * math.pi * self.EARTH_RADIUS / 360 / self.pixels_per_degree

    def set_meters_per_pixel(self, val):
        self.pixels_per_degree = 2 * math.pi * self.EARTH_RADIUS / 360 / val
        return val

    pixels_per_degree = property(get_pixels_per_degree, set_pixels_per_degree)
    meters_per_pixel = property(get_meters_per_pixel, set_meters_per_pixel)

    def is_scaled(self):
        return hasattr(self, '_pixels_per_degree')

    def project(self, coords):
        raise NotImplementedError

    def inverse_project(self, coords):
        # Not all projections can support this.
        raise NotImplementedError

    def auto_set_scale(self, extent_in, padding, width=None, height=None):
        # We need to choose a scale at which the data's bounding box,
        # once projected onto the map, will fit in the specified height
        # and/or width.  The catch is that we can't project until we
        # have a scale, so what we'll do is set a provisional scale,
        # project the bounding box onto the map, then adjust the scale
        # appropriately.  This way we don't need to know anything about
        # the projection.
        #
        # Projection subclasses are free to override this method with
        # something simpler that just solves for scale given the lat/lon
        # and x/y bounds.

        # We'll work large to minimize roundoff error.
        SCALE_FACTOR = 1000000.0
        self.pixels_per_degree = SCALE_FACTOR
        extent_out = extent_in.map(self.project)
        padding *= 2  # padding-per-edge -> padding-in-each-dimension
        try:
            if height:
                self.pixels_per_degree = pixels_per_lat = (
                    float(height - padding) /
                    extent_out.size().y * SCALE_FACTOR)
            if width:
                self.pixels_per_degree = (
                    float(width - padding) /
                    extent_out.size().x * SCALE_FACTOR)
                if height:
                    self.pixels_per_degree = min(self.pixels_per_degree,
                                                 pixels_per_lat)
        except ZeroDivisionError:
            raise ZeroDivisionError(
                'You need at least two data points for auto scaling. '
                'Try specifying the scale explicitly (or extent + '
                'height or width).')
        assert(self.pixels_per_degree > 0)


class EquirectangularProjection(Projection):
    # http://en.wikipedia.org/wiki/Equirectangular_projection
    def project(self, coord):
        x = coord.lon * self.pixels_per_degree
        y = -coord.lat * self.pixels_per_degree
        return Coordinate(x, y)

    def inverse_project(self, coord):
        lat = -coord.y / self.pixels_per_degree
        lon = coord.x / self.pixels_per_degree
        return LatLon(lat, lon)


class MercatorProjection(Projection):
    def set_pixels_per_degree(self, val):
        super(MercatorProjection, self).set_pixels_per_degree(val)
        self._pixels_per_radian = val * (180 / math.pi)
    pixels_per_degree = property(Projection.get_pixels_per_degree,
                                 set_pixels_per_degree)

    def project(self, coord):
        x = coord.lon * self.pixels_per_degree
        y = -self._pixels_per_radian * math.log(
            math.tan((math.pi/4 + math.pi/360 * coord.lat)))
        return Coordinate(x, y)

    def inverse_project(self, coord):
        lat = (360 / math.pi
               * math.atan(math.exp(-coord.y / self._pixels_per_radian)) - 90)
        lon = coord.x / self.pixels_per_degree
        return LatLon(lat, lon)


class Configuration(object):
    '''
    This object holds the settings for creating a heatmap as well as
    an iterator for the input data.

    Most of the command line processing is about settings and data, so
    the command line options are also processed with this object.
    This happens in two phases.

    First the settings are parsed and turned into more useful objects
    in set_from_options().  Command line flags go in, and the
    Configuration object is populated with the specified values and
    defaults.

    In the second phase, various other parameters are computed.  These
    are things we set automatically based on the other settings or on
    the data.  You can skip this if you set everything manually, but

    The idea is that someone could import this module, populate a
    Configuration instance manually, and run the process themselves.
    Where possible, this object contains instances, rather than option
    strings (e.g. for projection, kernel, colormap, etc).

    Every parameter is explained in the glossary dictionary, and only
    documented parameters are allowed.  Parameters default to None.
    '''

    glossary = {
        # Many of these are exactly the same as the command line option.
        # In those cases, the documentation is left blank.
        # Many have default values based on the command line defaults.
        'output': '',
        'width': '',
        'height': '',
        'margin': '',
        'shapes': 'unprojected iterable of shapes (Points and LineSegments)',
        'projection': 'Projection instance',
        'colormap': 'ColorMap instance',
        'decay': '',
        'kernel': 'kernel instance',
        'extent_in': 'extent in original space',
        'extent_out': 'extent in projected space',

        'background': '',
        'background_image': '',
        'background_brightness': '',

        # OpenStreetMap background tiles
        'osm': 'True/False; see command line options',
        'osm_base': '',
        'zoom': '',

        # These are for making an animation, ignored otherwise.
        'ffmpegopts': '',
        'keepframes': '',
        'frequency': '',
        'straggler_threshold': '',

        # We always instantiate an OptionParser in order to set up
        # default values.  You can use this OptionParser in your own
        # script, perhaps adding your own options.
        'optparser': 'OptionParser instance for command line processing',
    }

    _kernels = {'linear': LinearKernel,
                'gaussian': GaussianKernel, }
    _projections = {'equirectangular': EquirectangularProjection,
                    'mercator': MercatorProjection, }

    def __init__(self, use_defaults=True):
        for k in self.glossary.keys():
            setattr(self, k, None)

    def fill_missing(self):
        if not self.shapes:
            raise ValueError('no input specified')

        padding = self.margin + self.kernel.radius
        if not self.extent_in:
            logging.debug('reading input data')
            self.shapes = list(self.shapes)
            logging.debug('read %d shapes' % len(self.shapes))
            self.extent_in = Extent(shapes=self.shapes)

        if not self.projection.is_scaled():
            self.projection.auto_set_scale(self.extent_in, padding,
                                           self.width, self.height)
            if not (self.width or self.height or self.background_image):
                raise ValueError('You must specify width or'
                                 ' height or scale '
                                 'or background_image or'
                                 ' both osm and zoom.')

        if self.background_brightness is not None:
            if self.background_image:
                self.background_image = self.background_image.point(
                    lambda x: x * self.background_brightness)
                self.background_brightness = None   # idempotence
            else:
                logging.warning(
                    'background brightness specified, but no background image')

        if not self.extent_out:
            self.extent_out = self.extent_in.map(self.projection.project)
            self.extent_out.grow(padding)
        logging.info('input extent: %s' % str(self.extent_out.map(
            self.projection.inverse_project)))
        logging.info('output extent: %s' % str(self.extent_out))


class Point:
    def __init__(self, coord, weight=1.0):
        self.coord = coord
        self.weight = weight

    def __str__(self):
        return 'P(%s)' % str(self.coord)

    @staticmethod
    def general_distance(x, y):
        # assumes square units, which causes distortion in some projections
        return (x ** 2 + y ** 2) ** 0.5

    @property
    def extent(self):
        if not hasattr(self, '_extent'):
            self._extent = Extent(coords=(self.coord,))
        return self._extent

    # From a modularity standpoint, it would be reasonable to cache
    # distances, not heat values, and let the kernel cache the
    # distance to heat map, but this is substantially faster.
    heat_cache = {}

    @classmethod
    def _initialize_heat_cache(cls, kernel):
        cache = {}
        for x in range(kernel.radius + 1):
            for y in range(kernel.radius + 1):
                cache[(x, y)] = kernel.heat(cls.general_distance(x, y))
        cls.heat_cache[kernel] = cache

    def add_heat_to_matrix(self, matrix, kernel):
        if kernel not in Point.heat_cache:
            Point._initialize_heat_cache(kernel)
        cache = Point.heat_cache[kernel]
        x = int(self.coord.x)
        y = int(self.coord.y)
        for dx in range(-kernel.radius, kernel.radius + 1):
            for dy in range(-kernel.radius, kernel.radius + 1):
                matrix.add(Coordinate(x + dx, y + dy),
                           self.weight * cache[(abs(dx), abs(dy))])

    def map(self, func):
        return Point(func(self.coord), self.weight)


class LatLon(Coordinate):
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def get_lat(self):
        return self.y

    def set_lat(self, lat):
        self.y = lat

    def get_lon(self):
        return self.x

    def set_lon(self, lon):
        self.x = lon

    lat = property(get_lat, set_lat)
    lon = property(get_lon, set_lon)

    first = property(get_lat)
    second = property(get_lon)


class ImageMaker():
    def __init__(self, config):
        '''Each argument to the constructor should be a 4-tuple of (hue,
        saturaton, value, alpha), one to use for minimum data values and
        one for maximum.  Each should be in [0,1], however because hue is
        circular, you may specify hue in any range and it will be shifted
        into [0,1] as needed.  This is so you can wrap around the color
        wheel in either direction.'''
        self.config = config
        if config.background and not config.background_image:
            self.background = ImageColor.getrgb(config.background)
        else:
            self.background = None

    @staticmethod
    def _blend_pixels(a, b):
        # a is RGBA, b is RGB; we could write this more generically,
        # but why complicate things?
        alpha = a[3] / 255.0
        return tuple(
            map(lambda aa, bb: int(aa * alpha + bb * (1 - alpha)), a[:3], b))

    def make_image(self, matrix):
        extent = self.config.extent_out
        if not extent:
            extent = matrix.extent()
        extent.resize((self.config.width or 1) - 1,
                      (self.config.height or 1) - 1)
        size = extent.size()
        size.x = int(size.x) + 1
        size.y = int(size.y) + 1
        logging.info('saving image (%d x %d)' % (size.x, size.y))
        if self.background:
            img = Image.new('RGB', (size.x, size.y), self.background)
        else:
            img = Image.new('RGBA', (size.x, size.y))

        maxval = max(matrix.values())
        pixels = img.load()
        for (coord, val) in matrix.items():
            x = int(coord.x - extent.min.x)
            y = int(coord.y - extent.min.y)
            if extent.is_inside(coord):
                color = self.config.colormap.get(val / maxval)
                if self.background:
                    pixels[x, y] = ImageMaker._blend_pixels(color,
                                                            self.background)
                else:
                    pixels[x, y] = color
        if self.config.background_image:
            img = Image.composite(img, self.config.background_image,
                                  img.split()[3])
        return img


class Extent():
    def __init__(self, coords=None, shapes=None):
        if coords:
            coords = tuple(coords)  # if it's a generator, slurp them all
            self.min = coords[0].__class__(min(c.first for c in coords),
                                           min(c.second for c in coords))
            self.max = coords[0].__class__(max(c.first for c in coords),
                                           max(c.second for c in coords))
        elif shapes:
            self.from_shapes(shapes)
        else:
            raise ValueError('Extent must be initialized')

    def __str__(self):
        return '%s,%s,%s,%s' % (self.min.y, self.min.x, self.max.y, self.max.x)

    def update(self, other):
        '''grow this bounding box so that it includes the other'''
        self.min.x = min(self.min.x, other.min.x)
        self.min.y = min(self.min.y, other.min.y)
        self.max.x = max(self.max.x, other.max.x)
        self.max.y = max(self.max.y, other.max.y)

    def from_bounding_box(self, other):
        self.min = other.min.copy()
        self.max = other.max.copy()

    def from_shapes(self, shapes):
        shapes = iter(shapes)
        self.from_bounding_box(next(shapes).extent)
        for s in shapes:
            self.update(s.extent)

    def corners(self):
        return (self.min, self.max)

    def size(self):
        return self.max.__class__(self.max.x - self.min.x,
                                  self.max.y - self.min.y)

    def grow(self, pad):
        self.min.x -= pad
        self.min.y -= pad
        self.max.x += pad
        self.max.y += pad

    def resize(self, width=None, height=None):
        if width:
            self.max.x += float(width - self.size().x) / 2
            self.min.x = self.max.x - width
        if height:
            self.max.y += float(height - self.size().y) / 2
            self.min.y = self.max.y - height

    def is_inside(self, coord):
        return (coord.x >= self.min.x and coord.x <= self.max.x and
                coord.y >= self.min.y and coord.y <= self.max.y)

    def map(self, func):
        '''Returns a new Extent whose corners are a function of the
        corners of this one.  The expected use is to project a Extent
        onto a map.  For example: bbox_xy = bbox_ll.map(projector.project)'''
        return Extent(coords=(func(self.min), func(self.max)))


class ColorMap:
    DEFAULT_HSVA_MIN_STR = '02acfff00'
    DEFAULT_HSVA_MAX_STR = '02a00ffff'

    @staticmethod
    def _str_to_float(string, base=16, maxval=256):
        return float(int(string, base)) / maxval

    @staticmethod
    def str_to_hsva(string):
        '''
        Returns a 4-tuple of ints from a hex string color specification,
        such that AAABBCCDD becomes AAA, BB, CC, DD.  For example,
        str2hsva('06688bbff') returns (102, 136, 187, 255).  Note that
        the first number is 3 digits.
        '''
        if string.startswith('#'):
            # Leading "#" was once required, is now optional.
            string = string[1:]
        return tuple(ColorMap._str_to_float(s) for s in (string[0:3],
                                                         string[3:5],
                                                         string[5:7],
                                                         string[7:9]))

    def __init__(self, hsva_min=None, hsva_max=None, image=None, steps=256):
        '''
        Create a color map based on a progression in the specified
        range, or using pixels in a provided image.

        If supplied, hsva_min and hsva_max must each be a 4-tuple of
        (hue, saturation, value, alpha), where each is a float from
        0.0 to 1.0.  The gradient will be a linear progression from
        hsva_min to hsva_max, including both ends of the range.

        The optional steps argument specifies how many discrete steps
        there should be in the color gradient when using hsva_min
        and hsva_max.
        '''
        # TODO: do the interpolation in Lab space instead of HSV
        self.values = []
        if image:
            assert image.mode == 'RGBA', (
                'Gradient image must be RGBA.  Yours is %s.' % image.mode)
            num_rows = image.size[1]
            self.values = [image.getpixel((0, row)) for row in range(num_rows)]
            self.values.reverse()
        else:
            if not hsva_min:
                hsva_min = ColorMap.str_to_hsva(self.DEFAULT_HSVA_MIN_STR)
            if not hsva_max:
                hsva_max = ColorMap.str_to_hsva(self.DEFAULT_HSVA_MAX_STR)
            # Turn (h1,s1,v1,a1), (h2,s2,v2,a2) into (h2-h1,s2-s1,v2-v1,a2-a1)
            hsva_range = list(map(lambda min, max: max - min,
                                  hsva_min, hsva_max))
            for value in range(0, steps):
                hsva = list(map(
                    lambda range, min: value / float(steps - 1) * range + min,
                    hsva_range, hsva_min))
                hsva[0] = hsva[0] % 1  # in case hue is out of range
                rgba = tuple(
                    [int(x * 255) for x in
                        hsv_to_rgb(*hsva[0:3]) + (hsva[3],)])
                self.values.append(rgba)

    def get(self, floatval):
        return self.values[int(floatval * (len(self.values) - 1))]


class Matrix(defaultdict):
    '''An abstract sparse matrix, with data stored as {coord: value}.'''

    @staticmethod
    def matrix_factory(decay):
        # If decay is 0 or 1, we can accumulate as we go and save lots of
        # memory.
        if decay == 1.0:
            logging.info('creating a summing matrix')
            return SummingMatrix()
        elif decay == 0.0:
            logging.info('creating a maxing matrix')
            return MaxingMatrix()
        logging.info('creating an appending matrix')
        return AppendingMatrix(decay)

    def __init__(self, default_factory=float):
        self.default_factory = default_factory

    def add(self, coord, val):
        raise NotImplementedError

    def extent(self):
        return(Extent(coords=self.keys()))

    def finalized(self):
        return self


class SummingMatrix(Matrix):
    def add(self, coord, val):
        self[coord] += val


class MaxingMatrix(Matrix):
    def add(self, coord, val):
        self[coord] = max(val, self.get(coord, val))


class AppendingMatrix(Matrix):
    def __init__(self, decay):
        self.default_factory = list
        self.decay = decay

    def add(self, coord, val):
        self[coord].append(val)

    def finalized(self):
        logging.info('combining coincident points')
        m = Matrix()
        for (coord, values) in self.items():
            m[coord] = self.reduce(self.decay, values)
        return m

    @staticmethod
    def reduce(decay, values):
        '''
        Returns a weighted sum of the values, where weight N is
        pow(decay,N).  This means the largest value counts fully, but
        additional values have diminishing contributions. decay=0 makes
        the reduction equivalent to max(), which makes each data point
        visible, but says nothing about their relative magnitude.
        decay=1 makes this like sum(), which makes the relative
        magnitude of the points more visible, but could make smaller
        values hard to see.  Experiment with values between 0 and 1.
        Values outside that range will give weird results.
        '''
        # It would be nice to do this on the fly, while accumulating data, but
        # it needs to be insensitive to data order.
        weight = 1.0
        total = 0.0
        values.sort(reverse=True)
        for value in values:
            total += value * weight
            weight *= decay
        return total


def make_config(pts, hsva_min, hsva_max, bg):
    config = Configuration()
    config.background_image = bg
    (config.width, config.height) = config.background_image.size
    config.projection = config._projections['equirectangular']()
    config.radius = 2
    config.margin = 0
    padding = config.margin + config.radius
    config.kernel = config._kernels['linear'](config.radius)
    config.scale = None
    config.gradient = None
    config.decay = 0.3
    config.colormap = ColorMap(
        hsva_min=ColorMap.str_to_hsva(hsva_min),
        hsva_max=ColorMap.str_to_hsva(hsva_max))
    config.gpx = None
    config.extent_in = Extent(coords=(LatLon(-80., -180.), LatLon(80., 180.)))
    config.projection.auto_set_scale(config.extent_in, padding,
                                     config.width, config.height)
    config.extent_out = config.extent_in.map(config.projection.project)
    config.extent_out.grow(padding)
    config.osm = None
    config.shapes = pts
    config.background_brightness = None
    config.fill_missing()
    return config


def get_coord(addr):
    response = reader.city(addr)
    lat = response.location.latitude
    lng = response.location.longitude
    if lat and lng:
        return Point(LatLon(lat, lng), 1)
    return Point(LatLon(0.0, 0.0), 1)


def process_shapes(config, hook=None):
    matrix = Matrix.matrix_factory(config.decay)
    logging.info('processing data')
    for shape in config.shapes:
        shape = shape.map(config.projection.project)
        shape.add_heat_to_matrix(matrix, config.kernel)
        if hook:
            hook(matrix)
    return matrix


def get_heatmap(config):
    matrix = process_shapes(config).finalized()
    return ImageMaker(config).make_image(matrix)


def parse_cmdln():
    parser = argparse.ArgumentParser(
        description=__doc__,
        version=__version__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--in', dest='ip',
                        help='IP Database')
    parser.add_argument('-db', '--database', dest='db',
                        help='MaxMind GeoDB')
    parser.add_argument('-bg', '--background-image',
                        dest='bg', help='Image of the world.')
    parser.add_argument('-d', '--days', dest='days',
                        help='Number of days to access.', default=30, type=int)
    parser.add_argument('-m', '--min', dest='hsva_min',
                        help='Number of days to access.',
                        default=ColorMap.DEFAULT_HSVA_MIN_STR, type=str)
    parser.add_argument('-M', '--max', dest='hsva_max',
                        help='Number of days to access.',
                        default=ColorMap.DEFAULT_HSVA_MAX_STR, type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    options = parse_cmdln()

    reader = geoip2.database.Reader(options.db)
    world = Image.open(options.bg)
    ip = np.loadtxt(options.ip, dtype=str)

    pts = imap(get_coord, ip)

    config = make_config(pts, options.hsva_min, options.hsva_max, world)

    img = get_heatmap(config)

    img.save('./static/png/past' + str(options.days) + '.png')
