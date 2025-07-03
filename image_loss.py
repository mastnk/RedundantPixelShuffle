import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_perm( nElements, nGroups, roll=1 ):
    perm = list( range( nElements ) )
    perm = perm[-roll:] + perm[:-roll]
    perm = perm * nGroups

    perm = [ p + (i//nElements)*nElements for i, p in enumerate(perm) ]

    return perm

def group_roll( x ):
    pass

if( __name__ == "__main__" ):
    perm = _make_perm( 3, 4 )
    print( perm )

