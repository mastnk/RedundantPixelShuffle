import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseRedundantPixelShuffle():
    def __init__( self, kernel_size ):
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold( kernel_size )

    def __call__( self, x ):
        b, c, h, w = x.shape
        x = self.unfold( x ).view( b, c * self.kernel_size*self.kernel_size, h-(self.kernel_size-1), w-(self.kernel_size-1) )
        return x

class RedundantPixelShuffle():
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.fold = nn.Fold(
            output_size=None,  # Placeholder (set later explicitly)
            kernel_size=kernel_size
        )

    def __call__(self, x):
        b, c_k2, h_out, w_out = x.shape
        k = self.kernel_size
        c = c_k2 // (k * k)

        # Reshape to (b, c*k*k, L) where L = h_out * w_out
        x = x.view(b, c * k * k, -1)

        # Explicitly set the output size for folding
        self.fold.output_size = (h_out + k - 1, w_out + k - 1)

        # 1. Reconstruct the original image using fold
        out = self.fold(x)

        # 2. Compute the overlap count for averaging
        weight = torch.ones_like(x)
        overlap_count = self.fold(weight)
        out = out / overlap_count

        return out


if( __name__ == "__main__" ):
    b, c, h, w = 2, 3, 3, 4
    x = torch.tensor( list(range( b * c * h * w )), dtype=torch.float )
    x = x.view( b, c, h, w )

    k = 3
    IRPS = InverseRedundantPixelShuffle( k )
    RPS = RedundantPixelShuffle( k )

    y = IRPS( x )
    z = RPS( y )

    print( "x ch0" )
    print( x[0,0,:,:] )
    print( "x ch1" )
    print( x[0,1,:,:] )
    print( "x ch2" )
    print( x[0,2,:,:] )
    print()

    print( y.shape )
    for hh in range(y.shape[2]):
        for ww in range(y.shape[3]):
            print( f"y: {hh}, {ww}" )
            print( y[0,:,hh,ww] )
    print()

    print( "z ch0" )
    print( z[0,0,:,:] )
    print( "z ch1" )
    print( z[0,1,:,:] )
    print( "z ch2" )
    print( z[0,2,:,:] )
    print()

