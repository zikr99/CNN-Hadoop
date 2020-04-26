/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convnet;

import convnet.Util;
import convnet.Vol;
import java.io.Serializable;

/**
 *
 * @author zikr
 */
public class SoftmaxLayer extends FinalLayer implements Serializable {
    
    public int out_sx;
    public int out_sy;
    public int out_depth;
    public int num_inputs;
    public String layer_type;
    double[] es;
    
    public SoftmaxLayer(int pin_sx, int pin_sy, int pin_depth)
    {
        num_inputs = pin_sx*pin_sy*pin_depth;
	out_depth = num_inputs;
	out_sx = 1;
	out_sy = 1;
	layer_type = "softmax";
    }
	
    public Vol forward(Vol h, int o)
    {
        in_act = h;
			
	Vol f = new Vol(1, 1, out_depth, 0);
	double[] j = h.w;
	double k = h.w[0];

	for (int l = 1; l < this.out_depth; l++)
	{
            if (j[l] > k)
		k = j[l];
	}
			
	double[] n = Util.zeros(out_depth);
	double g = 0.0f;
			
        for (int l = 0; l < this.out_depth; l++)
	{
            double m = Math.exp(j[l] - k);
            g += m;
            n[l] = m;
	}
			
	for (int l = 0; l < this.out_depth; l++)
	{
            n[l] /= g;
            f.w[l] = n[l];
	}
			
	es = n;
	out_act = f;
	return out_act;
    }
		
    public double backward(int k)
    {
        Vol f = in_act;
	f.dw = Util.zeros(f.w.length);

	for (int h = 0; h < out_depth; h++)
	{
            int g = (h == k)?1:0;
            double j = -(g - this.es[h]);
				
            f.dw[h] = j;
	}
			
	return -Math.log(this.es[k]);
    }
}
