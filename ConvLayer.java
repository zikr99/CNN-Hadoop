/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convnet;

import java.util.List;
import java.util.ArrayList;
import convnet.Util;
import convnet.ParamsGrads;
import java.io.Serializable;
        
/**
 *
 * @author zikr
 */
public class ConvLayer extends Layer implements Serializable {
    
    public int out_depth;
    public int sx;
    public int in_depth;
    public int in_sx;
    public int in_sy;
    public int sy;
    public int stride;
    public int pad;
    public double l1_decay_mul;
    public double l2_decay_mul;
    public int out_sx;
    public int out_sy;
    public String layer_type;
    public double e;
    public List<Vol> filters;
    public Vol biases;
    
    public ConvLayer(int pin_sx, int pin_sy, int pin_depth,  
            int psx, int psy, int pstride, int ppad, int pout_depth, 
            double pl1_decay_mul, double pl2_decay_mul, double pe)
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
	out_depth = pout_depth;
		
	filters = new ArrayList<Vol>();

	for (int f = 0; f < out_depth; f++)
            filters.add(new Vol(sx, sy, in_depth));
	
        e = pe;
	biases = new Vol(1, 1, out_depth, e);
        
        l1_decay_mul = pl1_decay_mul;
	l2_decay_mul = pl2_decay_mul;
	layer_type = "conv";
    }
	
    public Vol forward(Vol h, int k)
    {
	in_act = h;
	Vol q = new Vol(out_sx, out_sy, out_depth, 0);
	int w = h.sx;
	int u = h.sy;
	int r = stride;

	for (int t = 0; t < out_depth; t++)
	{
            Vol s = filters.get(t);
            int n = -this.pad;
            int l = -this.pad;
				
            for (int m = 0; m < out_sy; l += r, m++)
            {
		n = -this.pad;

		for (int o = 0; o < out_sx; n += r, o++)
		{
                    double v = 0;

                    for (int e = 0; e < s.sy; e++)
                    {
			int i = l + e;

			for (int g = 0; g < s.sx; g++)
                        {
                            int j = n + g;
								
                            if ((i >= 0) && (i < u) && (j >= 0) && (j < w))
                            {
				for (int p = 0; p < s.depth; p++)
				{
                                    v += s.w[((s.sx*e) + g)*s.depth + p]*h.w[((w*i) + j)*h.depth + p];
				}
                            }
                        }
                    }
						
                    v += biases.w[t];
                    q.set(o, m, t, v);
		}
            }
        }
			
	out_act = q;
	return out_act;
    }
		
    public void backward()
    {
	Vol i = in_act;
	i.dw = Util.zeros(i.w.length);
	int w = i.sx;
	int v = i.sy;
        int q = this.stride;
			
	for (int t = 0; t < out_depth; t++)
	{
            Vol r = filters.get(t);
            int n = -this.pad;
            int l = -this.pad;

            for (int m = 0; m < out_sy; l += q, m++)
            {
		n = -this.pad;

		for (int o = 0; o < out_sx; n += q, o++)
		{
                    double e = out_act.get_grad(o, m, t);

                    for (int g = 0; g < r.sy; g++)
                    {
			int j = l + g;

			for (int h = 0; h < r.sx; h++)
			{
                            int k = n + h;
								
                            if ((j >= 0) && (j < v) && (k >= 0) && (k < w))
                            {
				for (int p = 0; p < r.depth; p++)
				{
                                    int u = ((w*j) + k)*i.depth + p;
                                    int s = ((r.sx*g) + h)*r.depth + p;
                                    r.dw[s] += i.w[u]*e;
                                    i.dw[u] += r.w[s]*e;
				}
                            }
			}
                    }
						
                    biases.dw[t] += e;
		}
            }
        }
    }
		
    public List<ParamsGrads> getParamsAndGrads()
    {
	List<ParamsGrads> e = new ArrayList<ParamsGrads>();

        for (int f = 0; f < this.out_depth; f++)
        {
            ParamsGrads p = new ParamsGrads();
            
            p.params = ((Vol)this.filters.get(f)).w;
            p.grads = ((Vol)this.filters.get(f)).dw;
            p.l1_decay_mul = this.l1_decay_mul;
            p.l2_decay_mul = this.l2_decay_mul;
            
            e.add(p);
        }
        
        ParamsGrads b = new ParamsGrads();
            
        b.params = this.biases.w;
        b.grads = this.biases.dw;
        b.l1_decay_mul = 0.0;
        b.l2_decay_mul = 0.0;
      
        e.add(b);
        
        return e;
    }
};
