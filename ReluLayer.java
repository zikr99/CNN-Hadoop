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
import convnet.ConvLayer;
import java.io.Serializable;

/**
 *
 * @author zikr
 */
public class ReluLayer extends Layer implements Serializable {
    
    public int out_sx;
    public int out_sy;
    public int out_depth;
    public String layer_type;
    
    public ReluLayer(ConvLayer h)
    {
        out_sx = h.in_sx;
	out_sy = h.in_sy;
	out_depth = h.in_depth;
	layer_type = "relu";
    }
	
    public Vol forward(Vol j, int l)
    {
        in_act = j;
        Vol h = j.clone();
        int m = j.w.length;
	double[] n = h.w;

	for (int k = 0; k < m; k++)
	{
            if (n[k] < 0)
		n[k] = 0;
	}
			
	out_act = h;
	return this.out_act;
    }
		
    public void backward()
    {
        Vol j = in_act;
	Vol h = this.out_act;
	int l = j.w.length;
        j.dw = Util.zeros(l);

	for (int k = 0; k < l; k++)
	{
            if (h.w[k] <= 0)
		j.dw[k] = 0;
            else
		j.dw[k] = h.dw[k];
	}
    }
};
