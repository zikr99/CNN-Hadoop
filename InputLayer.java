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
public class InputLayer extends Layer implements Serializable {
    
    public int out_sx;
    public int out_sy;
    public int out_depth;
    public String layer_type;
    
    public InputLayer(int pout_sx, int pout_sy, int pout_depth)
    {
        out_sx = pout_sx;
	out_sy = pout_sy;
        out_depth = pout_depth;
	layer_type = "input";
    }
	
    public Vol forward(Vol e, int f)
    {
        in_act = e;
	out_act = e;
			
        return this.out_act;
    }
    
    public void backward()
    {
    }
}
