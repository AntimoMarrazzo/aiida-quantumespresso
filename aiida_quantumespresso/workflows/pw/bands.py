# -*- coding: utf-8 -*-
from aiida.common.extendeddicts import AttributeDict
from aiida.orm.calculation import JobCalculation
from aiida.orm.data.base import Str, Bool
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.bands import BandsData
from aiida.orm.group import Group
from aiida.orm.utils import WorkflowFactory
from aiida.work.workchain import WorkChain, ToContext, if_
from aiida.work import workfunction
from aiida_quantumespresso.utils.mapping import prepare_process_inputs


PwBaseWorkChain = WorkflowFactory('quantumespresso.pw.base')
PwRelaxWorkChain = WorkflowFactory('quantumespresso.pw.relax')


class PwBandsWorkChain(WorkChain):
    """Workchain to compute a band structure for a given structure using Quantum ESPRESSO pw.x"""

    @classmethod
    def define(cls, spec):
        super(PwBandsWorkChain, cls).define(spec)
        spec.expose_inputs(PwRelaxWorkChain, namespace='relax', exclude=('structure',))
        spec.expose_inputs(PwBaseWorkChain, namespace='scf', exclude=('structure', 'kpoints'))
        spec.expose_inputs(PwBaseWorkChain, namespace='bands', exclude=('structure',))
        spec.input('structure', valid_type=StructureData)
        spec.input('pseudo_family', valid_type=Str)
        spec.input('kpoints_distance', valid_type=Float, default=Float(0.2))
        spec.input('kpoints_distance_bands', valid_type=Float, default=Float(0.2))
        spec.input('kpoints', valid_type=KpointsData, required=False)
        spec.input('vdw_table', valid_type=SinglefileData, required=False)
        spec.input('parameters', valid_type=ParameterData)
        spec.input('settings', valid_type=ParameterData, required=False)
        spec.input('options', valid_type=ParameterData, required=False)
        spec.input('automatic_parallelization', valid_type=ParameterData, required=False)
        # Use workchain_options to control the wc behaviour, see the setup step for 
        # a list of valid keywords.
        spec.input('workchain_options',  valid_type=ParameterData, required=False)
        spec.input('clean_workdir', valid_type=Bool, default=Bool(False))
        spec.input('group', valid_type=Str, required=False)
        spec.outline(
            cls.setup,
            if_(cls.should_do_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            cls.run_seekpath,
            cls.run_scf,
            cls.inspect_scf,
            cls.run_bands,
            cls.inspect_bands,
            cls.results,
        )
        spec.exit_code(401, 'ERROR_SUB_PROCESS_FAILED_RELAX',
            message='the PwRelaxWorkChain sub process failed')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_SCF',
            message='the scf PwBasexWorkChain sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_BANDS',
            message='the bands PwBasexWorkChain sub process failed')
        spec.output('primitive_structure', valid_type=StructureData)
        spec.output('seekpath_parameters', valid_type=ParameterData)
        spec.output('scf_parameters', valid_type=ParameterData)
        spec.output('band_parameters', valid_type=ParameterData)
        spec.output('band_structure', valid_type=BandsData)

    def setup(self):
        """
        Initialize context variables that are used during the logical flow of the BaseRestartWorkChain
        """
        self.ctx.inputs = AttributeDict({
            'code': self.inputs.code,
            'pseudo_family': self.inputs.pseudo_family,
            'parameters': self.inputs.parameters.get_dict(),
        })

    def validate_inputs(self):
        """
        Validate inputs that may depend on each other
        """
        if not any([key in self.inputs for key in ['options', 'automatic_parallelization']]):
            self.abort_nowait('you have to specify either the options or automatic_parallelization input')
            return

        # Add the van der Waals kernel table file if specified
        if 'vdw_table' in self.inputs:
            self.ctx.inputs.vdw_table = self.inputs.vdw_table
            self.inputs.relax['vdw_table'] = self.inputs.vdw_table

        # Set the correct relaxation scheme in the input parameters
        if 'CONTROL' not in self.ctx.inputs.parameters:
            self.ctx.inputs.parameters['CONTROL'] = {}

        if 'settings' in self.inputs:
            self.ctx.inputs.settings = self.inputs.settings

        # If options set, add it to the default inputs
        if 'options' in self.inputs:
            self.ctx.inputs.options = self.inputs.options

        # If automatic parallelization was set, add it to the default inputs
        if 'automatic_parallelization' in self.inputs:
            self.ctx.inputs.automatic_parallelization = self.inputs.automatic_parallelization

        if 'workchain_options' in self.inputs:
            wc_options = self.inputs['workchain_options'].get_dict()
        else:
            wc_options = {}
       
        # Setting number of bands for the band structure calculations
        # defined as a multiplicative factore wrt to the number of occupied states
        # e.g. 1.2 means 20% more or at least 4 more as in QE
        try:
            self.ctx.num_bands_factor = wc_options.pop('num_bands_factor')
        except KeyError:
            self.ctx.num_bands_factor = 1.2    # 20% more than num occupied states, as in QE
       
        if wc_options:
            #Checking that all options given in input are recognised
            self.abort_nowait('Unknown variable passed inside workchain_options: {}'.format(
                              ', '.join(wc_options.keys())
                              ))
        return
       
    def validate_inputs(self):
        """
        Validate inputs that may depend on each other
        """
        if not any([key in self.inputs for key in ['options', 'automatic_parallelization']]):
            self.abort_nowait('you have to specify either the options or automatic_parallelization input')
            return
        """Define the current structure in the context to be the input structure."""
        self.ctx.current_structure = self.inputs.structure

    def should_do_relax(self):
        """If the 'relax' input namespace was specified, we relax the input structure."""
        return 'relax' in self.inputs

    def run_relax(self):
        """Run the PwRelaxWorkChain to run a relax PwCalculation."""
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        inputs.structure = self.ctx.current_structure

        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report('launching PwRelaxWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_relax=running)

    def inspect_relax(self):
        """Verify that the PwRelaxWorkChain finished successfully."""
        workchain = self.ctx.workchain_relax

        if not workchain.is_finished_ok:
            self.report('PwRelaxWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX
        else:
            self.ctx.current_structure = workchain.out.output_structure

    def run_seekpath(self):
        """
        Run the relaxed structure through SeeKPath to get the new primitive structure, just in case
        the symmetry of the cell changed in the cell relaxation step
        """
        if 'kpoints_distance' in self.inputs.bands:
            seekpath_parameters = ParameterData(dict={
                'reference_distance': self.inputs.bands.kpoints_distance.value
            })
        else:
            seekpath_parameters = ParameterData(dict={})

        result = seekpath_structure_analysis(self.ctx.current_structure, seekpath_parameters)
        self.ctx.current_structure = result['primitive_structure']
        self.ctx.kpoints_path = result['explicit_kpoints']

        self.out('primitive_structure', result['primitive_structure'])
        self.out('seekpath_parameters', result['parameters'])

    def run_scf(self):
        """Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure"""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.structure = self.ctx.current_structure
        inputs.parameters = inputs.parameters.get_dict()
        inputs.parameters.setdefault('CONTROL', {})
        inputs.parameters['CONTROL']['calculation'] = 'scf'

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode'.format(running.pk, 'scf'))

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Verify that the PwBaseWorkChain for the scf run finished successfully."""
        workchain = self.ctx.workchain_bands

        inputs = self.ctx.inputs
        restart_mode = 'restart'
        calculation_mode = 'bands'
       
        scf_out_dict = self.ctx.workchain_scf.out.output_parameters.get_dict()
        num_elec = int(scf_out_dict['number_of_electrons'])        
        num_spin = int(scf_out_dict['number_of_spin_components'])
        # Set the correct pw.x input parameters
        inputs['parameters']['CONTROL']['restart_mode'] = restart_mode
        inputs['parameters']['CONTROL']['calculation'] = calculation_mode
        # This gives the same results also with noncollinear calcs
        # e.g. with 8 electrons, no spinors and a factor of 1.5 you compute 6 bands
        #      with spinors you compute 12 bands (still 50% more than the occupied states)
        # As in QE, we add at least 4 (8 with spinors) additional bands 
        inputs['parameters']['SYSTEM']['nbnd'] = max(int(0.5 * num_elec * num_spin * self.ctx.num_bands_factor),
                                                     int(num_elec * num_spin * 0.5) + 4*num_spin)
        
        # Tell the plugin to retrieve the bands
        settings = inputs['settings'].get_dict()

        if 'kpoints' in self.inputs:
            inputs.kpoints = self.inputs.kpoints
        if not workchain.is_finished_ok:
            self.report('scf PwBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF
        else:
            self.ctx.current_folder = workchain.out.remote_folder

    def run_bands(self):
        """Run the PwBaseWorkChain in bands mode along the path of high-symmetry determined by Seekpath."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='bands'))
        inputs.parameters = inputs.parameters.get_dict()
        inputs.parameters.setdefault('CONTROL', {})
        inputs.parameters['CONTROL']['restart_mode'] = 'restart'
        inputs.parameters['CONTROL']['calculation'] = 'bands'

        if 'kpoints' not in self.inputs.bands:
            inputs.kpoints = self.ctx.kpoints_path

        inputs.structure = self.ctx.current_structure
        inputs.parent_folder = self.ctx.current_folder

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode'.format(running.pk, 'bands'))

        return ToContext(workchain_bands=running)

    def inspect_bands(self):
        """Verify that the PwBaseWorkChain for the bands run finished successfully."""
        workchain = self.ctx.workchain_bands

        if not workchain.is_finished_ok:
            self.report('bands PwBaseWorkChain failed with exit status {}'.format(workchain.exit_status))
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain."""
        self.report('workchain succesfully completed')
        self.out('scf_parameters', self.ctx.workchain_scf.out.output_parameters)
        self.out('band_parameters', self.ctx.workchain_bands.out.output_parameters)
        self.out('band_structure', self.ctx.workchain_bands.out.output_band)

        if 'group' in self.inputs:
            output_band = self.ctx.workchain_bands.out.output_band
            group, _ = Group.get_or_create(name=self.inputs.group.value)
            group.add_nodes(output_band)
            self.report("storing the output_band<{}> in the group '{}'"
                .format(output_band.pk, self.inputs.group.value))

    def on_terminated(self):
        """
        If the clean_workdir input was set to True, recursively collect all called Calculations by
        ourselves and our called descendants, and clean the remote folder for the JobCalculation instances
        """
        super(PwBandsWorkChain, self).on_terminated()

        if self.inputs.clean_workdir.value is False:
            self.report('remote folders will not be cleaned')
            return

        cleaned_calcs = []

        for called_descendant in self.calc.called_descendants:
            if isinstance(called_descendant, JobCalculation):
                try:
                    called_descendant.out.remote_folder._clean()
                    cleaned_calcs.append(called_descendant.pk)
                except (IOError, OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report('cleaned remote folders of calculations: {}'.format(' '.join(map(str, cleaned_calcs))))


@workfunction
def seekpath_structure_analysis(structure, parameters):
    """
    This workfunction will take a structure and pass it through SeeKpath to get the
    primitive cell and the path of high symmetry k-points through its Brillouin zone.
    Note that the returned primitive cell may differ from the original structure in
    which case the k-points are only congruent with the primitive cell.
    """
    seekpath_info = get_path(structure)
    explicit_path = get_explicit_k_path(structure)

    primitive_structure = seekpath_info.pop('primitive_structure')
    conv_structure = seekpath_info.pop('conv_structure')
    parameters = ParameterData(dict=seekpath_info)

    result = {
        'parameters': parameters,
        'conv_structure': conv_structure,
        'primitive_structure': primitive_structure,
        'explicit_kpoints_path': explicit_path['explicit_kpoints'],
    }

    return result
    from aiida.tools import get_explicit_kpoints_path
    return get_explicit_kpoints_path(structure, **parameters.get_dict())
