import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_perm( nElements, nGroups, roll=1 ):
    perm = list( range( nElements ) )
    perm = perm[-roll:] + perm[:-roll]
    perm = perm * nGroups

    perm = [ p + (i//nElements)*nElements for i, p in enumerate(perm) ]

    return perm

def channel_group_roll( x, nElements, nGroups, roll=1 ):
    perm = _make_perm( nElements, nGroups )
    return x[:, perm, :, :]

if( __name__ == "__main__" ):
    perm = _make_perm( 3, 4 )
    print( perm )

    x = torch.tensor( list(range(3*4)), dtype=torch.float ).view( 1, 3*4, 1, 1 )
    print(x)
    x = channel_group_roll( x, 3, 4 )
    print(x)

