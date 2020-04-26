/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convnet;

import java.util.ArrayList;
import java.util.List;
import convnet.Util;
import convnet.Vol;
import java.io.Serializable;

/**
 *
 * @author zikr
 */
public class PoolLayer extends Layer implements Serializable {
    
    public int out_depth;
    public int sx;
    public int in_depth;
    public int in_sx;
    public int in_sy;
    public int sy;
    public int stride;
    public int pad;
    public int out_sx;
    public int out_sy;
    public String layer_type;
    public int e;
    public double[] switchx;
    public double[] switchy;
    
    public PoolLayer(int pin_sx, int pin_sy, int pin_depth, 
            int psx, int psy, int pstride, int ppad)
    {
        in_sx = pin_sx;
	in_sy = pin_sy;
        in_depth = pin_depth;
        sx = psx;
	sy = psy;
        stride = pstride;
	pad = ppad;
        out_sx = (int)Math.floor((in_sx + this.pad*2 - sx)/stride + 1);
	out_sy = (int)Math.floor((in_sy + this.pad*2 - sy)/stride + 1);
	out_depth = in_depth;
	switchx = Util.zeros(out_sx*out_sy*out_depth);
	switchy = Util.zeros(out_sx*out_sy*out_depth);
        layer_type = "pool";
    }
	
    public Vol forward(Vol l, int u)
    {
        in_act = l;
	Vol h = new Vol(out_sx, out_sy, out_depth, 0);

	int i = 0;

	for (int p = 0; p < out_depth; p++)
	{
            int s = -this.pad;
            int q = -this.pad;

            for (int e = 0; e < out_sx; s += stride, e++)
            {
		q = -this.pad;

                for (int w = 0; w < out_sy; q += stride, w++)
		{
                    double r = -99999.0f;
                    int o = -1, k = -1;

                    for (int m = 0; m < sx; m++)
                    {
			for (int j = 0; j < sy; j++)
			{
                            int f = q + j;
                            int g = s + m;
								
                            if ((f >= 0) && (f < l.sy) && (g >= 0) && (g < l.sx))
                            {
				double t = l.get(g, f, p);

				if (t > r)
				{
                                    r = t;
                                    o = g;
                                    k = f;
				}
                            }
                        }
                    }
						
                    switchx[i] = o;
                    switchy[i] = k;
                    i++;
						
                    h.set(e, w, p, r);
                }
            }
        }
			
        out_act = h;
	return out_act;
    }
		
    public void backward()
    {
        Vol h = this.in_act;
	h.dw = Util.zeros(h.w.length);
			
	Vol f = this.out_act;
	int g = 0;
			
	for (int j = 0; j < this.out_depth; j++)
	{
            int l = -this.pad;
            int k = -this.pad;

            for (int e = 0; e < out_sx; l += stride, e++)
            {
		k = -this.pad;

		for (int m = 0; m < out_sy; k += stride, m++)
		{
                    double i = this.out_act.get_grad(e, m, j);
                    h.add_grad((int)switchx[g], (int)switchy[g], j, i);
                    g++;
		}
            }
        }
    }
}
