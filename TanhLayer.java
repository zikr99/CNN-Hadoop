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
public class TanhLayer extends Layer implements Serializable {
    
    public int out_sx;
    public int out_sy;
    public int out_depth;
    public String layer_type;
    
    protected double c(double h)
    {
	double i = Math.exp(2*h);
	return (i - 1)/(i + 1);
    }
    
    public TanhLayer(ConvLayer h)
    {
        out_sx = h.in_sx;
	out_sy = h.in_sy;
	out_depth = h.in_depth;
	layer_type = "tanh";
    }
	
    public Vol forward(Vol j, int l)
    {
        in_act = j;
	Vol h = j.cloneAndZero();
	int m = j.w.length;
			
	for (int k = 0; k < m; k++)
            h.w[k] = c(j.w[k]);
			
	out_act = h;
	return out_act;
    }
		
    public void backward()
    {
        Vol j = in_act;
	Vol h = out_act;
	int m = j.w.length;
	j.dw = Util.zeros(m);
			
	for (int k = 0; k < m; k++)
	{
            double l = h.w[k];
            j.dw[k] = (1 - l*l)*h.dw[k];
	}
    }
}
