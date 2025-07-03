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

def huber( x, delta ):
    x = x.abs()
    return (x.square()/(2*delta)).clip(None, delta/2) + F.relu( x - delta )

def spatial_diff( x ):
    xh = x[:,:,:,1:] - x[:,:,:,:-1]
    xv = x[:,:,1:,:] - x[:,:,:-1,:]
    return xh, xv

def channel_diff( x, nChannels ):
    nGroups = x.shape[1] // nChannels
    return x - channel_group_roll( x, nChannels, nGroups, 1 )


class ImageLoss(nn.Module):
    def __init__(self, nChannels=3, alpha=10, beta=5, delta=5/255):
        super(ImageLoss, self).__init__()

        self.nChannels = nChannels
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def forward( self, x, y ):
        loss = huber(x-y, self.delta).mean( dim=(1,2,3) )

        if( self.alpha > 0 ):
            xh, xv = spatial_diff( x )
            yh, yv = spatial_diff( y )
            loss = loss + self.alpha * ( huber(xh-yh, self.delta).mean( dim=(1,2,3) ) + huber(xh-yh, self.delta).mean( dim=(1,2,3) ) )

        if( self.beta > 0 ):
            xc = channel_diff( x, self.nChannels )
            yc = channel_diff( y, self.nChannels )
            loss = loss + self.beta * huber( xc-yc, self.delta).mean( dim=(1,2,3) )

        return loss


if( __name__ == "__main__" ):
    perm = _make_perm( 3, 4 )
    print( perm )

    x = torch.tensor( list(range(3*4)), dtype=torch.float ).view( 1, 3*4, 1, 1 )
    print(x)
    x = channel_group_roll( x, 3, 4 )
    print(x)

    b, c, h, w = 1, 3, 4, 3
    k = 5

    cri = ImageLoss( nChannels=3 )
    x = torch.randn( (b, c*k*k, h, w ), dtype=torch.float )
    y = torch.randn( (b, c*k*k, h, w ), dtype=torch.float )

    loss = cri(x, y )
    print( loss )

