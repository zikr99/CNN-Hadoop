/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package convnet;

import java.util.List;
import java.util.ArrayList;
import convnet.Vol;
import convnet.ParamsGrads;

/**
 *
 * @author zikr
 */
public class Layer {
    
    public Vol in_act;
    public Vol out_act;
    
    public Vol forward(Vol a, int b)
    {
        return new Vol(1, 1, 1, 0.0f);
    }
    
    public void backward()
    {
        
    }
    
    public List<ParamsGrads> getParamsAndGrads()
    {
        return new ArrayList<ParamsGrads>();
    }
}
