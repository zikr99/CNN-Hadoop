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
public class FullyConnLayer extends Layer implements Serializable {
    
    public int out_depth;
    public int sx;
    public int in_depth;
    public double l1_decay_mul;
    public double l2_decay_mul;
    public int num_inputs;
    public int out_sx;
    public int out_sy;
    public String layer_type;
    public double e;
    public List<Vol> filters;
    public Vol biases;
    
    public FullyConnLayer(int pin_sx, int pin_sy, int pin_depth, int pout_depth,  
            double pl1_decay_mul, double pl2_decay_mul, double pe)
    {
	num_inputs = pin_sx*pin_sy*pin_depth;
	out_sx = 1;
	out_sy = 1;
        out_depth = pout_depth;
		
	filters = new ArrayList<Vol>();
		
	for (int f = 0; f < this.out_depth; f++)
            filters.add(new Vol(1, 1, num_inputs));
	
        e = pe;
	biases = new Vol(1, 1, this.out_depth, e);
        
        layer_type = "fc";
    }
	
    public Vol forward(Vol h, int l)
    {
        in_act = h;
	Vol f = new Vol(1, 1, out_depth, 0);
	double[] k = h.w;

	for (int j = 0; j < out_depth; j++)
	{
            double g = 0.0f;
            double[] e = filters.get(j).w;
				
            for (int m = 0; m < num_inputs; m++)
		g += k[m]*e[m];	
				
            g += biases.w[j];
            f.w[j] = g;
	}
			
	out_act = f;
	return out_act;
    }
		
    public void backward()
    {
        Vol e = in_act;
	e.dw = Util.zeros(e.w.length);
			
	for (int f = 0; f < out_depth; f++)
	{
            Vol h = filters.get(f);
            double g = out_act.dw[f];
				
            for (int j = 0; j < num_inputs; j++)
            {
		e.dw[j] += h.w[j]*g;
		h.dw[j] += e.w[j]*g;
            }
				
            biases.dw[f] += g;
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
