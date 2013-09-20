# -*- coding:utf-8 -*-

# Haversine formula example in Python
# Author: Wayne Dyck

import math

def distance(origin, destination):
    """
    >>> import haversine
    >>> seattle = [47.621800, -122.350326] # 47° 36′ N, 122° 20′ W
    >>> olympia = [47.041917, -122.893766] # 47°2′33″N 122°53′35″W
    >>> distance(seattle, olympia)
    76.386615799548693    
    """
   
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

def lrange(iterable, i=0):
    return range(i, len(iterable))

def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate    
