/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convnet;

import convnet.Util;
import convnet.Vol;
import convnet.ConvLayer;
import java.io.Serializable;

/**
 *
 * @author zikr
 */
public class SigmoidLayer extends Layer implements Serializable {
    
    public int out_depth;
    public int out_sx;
    public int out_sy;
    public String layer_type;
    
    public SigmoidLayer(ConvLayer h)
    {
        out_sx = h.in_sx;
	out_sy = h.in_sy;
	out_depth = h.in_depth;
	layer_type = "sigmoid";
    }
	
    public Vol forward(Vol j, int m)
    {
        in_act = j;
	Vol h = j.cloneAndZero();
	int n = j.w.length;
	double[] o = h.w;
	double[] l = j.w;

	for (int k = 0; k < n; k++)
            o[k] = 1/(1 + Math.exp(-l[k]));
			
	out_act = h;
	return out_act;
    }
		
    public void backward()
    {
        Vol j = in_act;
	Vol h = this.out_act;
	int m = j.w.length;
	j.dw = Util.zeros(m);
			
	for (int k = 0; k < m; k++)
	{
            double l = h.w[k];
            j.dw[k] = l*(1 - l)*h.dw[k];
	}
    }
};
