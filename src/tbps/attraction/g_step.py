"""在原G K+1机制中替换Global；保留原属性Local、前轮Retain及严格投影。"""
from contextlib import contextmanager
from unittest.mock import patch
from .objectives import scalars,require

@contextmanager
def attraction_g(serial,profile,clean_image,partner_text,records):
    old_attribute,old_joint=serial.attribute_losses,serial.joint_losses
    def replace_global(terms,feature,phase):
        clean=clean_image.detach().to(feature);partner=partner_text.detach().to(feature)
        values=profile.terms(feature,clean,partner)
        original=terms['global'];terms['global']=values['loss'].mean()
        terms['total']=terms['total']-original+terms['global']
        records.append({'phase':phase,'old_global_diagnostic':float(original.detach()),'image_feature':feature.detach().cpu().reshape(-1).tolist(),**scalars(values)})
        return terms
    def attribute(feature,z,local,**kwargs):return replace_global(old_attribute(feature,z,local,**kwargs),feature,'attribute')
    def joint(feature,z,local,has_local):return replace_global(old_joint(feature,z,local,has_local),feature,'joint')
    try:
        with patch.object(serial,'attribute_losses',attribute),patch.object(serial,'joint_losses',joint):yield
    finally:
        require(serial.attribute_losses is old_attribute and serial.joint_losses is old_joint,'restore original G bindings')
