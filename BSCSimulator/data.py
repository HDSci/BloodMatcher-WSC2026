"""Module for data input (and output) handling."""

import datetime
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import yaml
from scipy.stats._distn_infrastructure import rv_discrete_frozen

from .bloodgroups import bloodgroup_frequency
from .simulator import SimulationManager
from .util import list_of_permutations

logger = logging.getLogger(__name__)


class DataIO:
    
    def __init__(self, output_root: str, tuning=False) -> None:
        self.output_root_folder = output_root
        self.is_tuning = tuning
        if self.is_tuning:
            self.output_root_folder = os.path.realpath(self.output_root_folder)
        self.matching_rules: dict = None
        self.immunogenicities: pd.DataFrame = None
        self.ab_data: pd.DataFrame = None
        self.init_age_dist: pd.DataFrame = None
        self.location_data = {}
        self.donor_data = {}
        self.patient_data = {}
        self.manufacturing_routing_data = {}
        self.dummy_data: pd.DataFrame = None
        self._supply_names = {}
        self.mismatch_parameters = {}

    def load_supply(self, filename: str = 'data/locations/supply.yml') -> None:
        """Loads supply data

        :param str filename: Filepath to YAML file, defaults to 'data/locations/supply.yml'
        """
        with open(filename) as f:
            a = yaml.load(f, Loader=yaml.FullLoader)
        self.location_data.update({'supply': a})
        for name, d in a.items():
            id = d.get('UniqueID')
            supply_data: dict = d.get('Supply')
            black_pop_ratio = supply_data.get('DonorEthnicities').get('AfricanDonors')
            config_file = supply_data.get('BloodGroupDistribution')
            donor_data = population_phenotype(config_file, black_pop_ratio)
            self.donor_data.update({id: donor_data})
            self._supply_names.update({id: name})
            donations_by_day_dist = supply_data.get('DonationsbyDayDistributions')
            donations_dist = supply_data.get('DonationsDistribution')
            if donations_by_day_dist is not None:
                distributions = [
                    create_distribution_object(donations_by_day_dist, {'name': str(name) + '_DonationsbyDay',
                                                                        'mean': val_or_dict,
                                                                        'support': list(val_or_dict.keys()) if type(val_or_dict) is dict else [int(val_or_dict)],
                                                                        'probabilities': list(val_or_dict.values()) if type(val_or_dict) is dict else [1.],}
                    ) for val_or_dict in supply_data.get('DonationsbyDayDistributionsValues', ) 
                ]
                supply_data.update({'Donations': distributions})
            elif donations_dist is not None:
                dist_params = {'name': str(name) + '_Donations',
                            'mean': supply_data.get('DonationsDistributionValue'),
                            'support': list(supply_data.get('DonationsDistributionValues', {}).keys()),
                            'probabilities': list(supply_data.get('DonationsDistributionValues', {}).values()),}
                distribution = create_distribution_object(donations_dist, dist_params)
                supply_data.update({'Donations': distribution})
    
    def load_demand(self, filename: str = 'data/locations/demand.yml') -> None:
        """Loads demand data

        :param str filename: Filepath to YAML file, defaults to 'data/locations/demand.yml'
        """
        with open(filename) as f:
            a: dict[str, dict] = yaml.load(f, Loader=yaml.FullLoader)
        self.location_data.update({'demand': a})
        for name, d in a.items():
            id = d.get('UniqueID')
            patient_groups: dict[str, dict] = d.get('PatientGroups')
            patient_data = {}
            for grp, data in patient_groups.items():
                if data['Type'] == 'Appointments':
                    config_file = data['BloodGroupDistribution']
                    black_pop_ratio = data.get('PatientEthnicities').get('AfricanPatients')
                    patient_grp_data = population_phenotype(config_file, black_pop_ratio)
                    patient_grp_ab_data = self.ab_data.copy().loc[[grp],:] if grp in self.ab_data.index else self.ab_data.copy().loc[['default'], :]
                    units_per_appointment_dist = data.get('UnitsperAppointmentDistribution')
                    if units_per_appointment_dist is not None:
                        dist_params = {'name': str(name) + '_UnitsperAppointment',
                                       'mean': data.get('UnitsperAppointment'),
                                       'support': list(data.get('UnitsperAppointmentDistributionValues', {}).keys()),
                                       'probabilities': list(data.get('UnitsperAppointmentDistributionValues', {}).values()), }
                        distribution = create_distribution_object(units_per_appointment_dist, dist_params)
                        data.update({'UnitsperAppointment': distribution})
                    appointments_dist = data.get('AppointmentsDistribution')
                    appointments_by_day_dist = data.get('AppointmentsbyDayDistributions')
                    if appointments_by_day_dist is not None:
                        distributions = [
                            create_distribution_object(appointments_by_day_dist, {'name': str(name) + '_AppointmentsbyDay',
                                                                                    'mean': val_or_dict,
                                                                                    'support': list(val_or_dict.keys()) if type(val_or_dict) is dict else [int(val_or_dict)],
                                                                                    'probabilities': list(val_or_dict.values()) if type(val_or_dict) is dict else [1.],}
                            ) for val_or_dict in data.get('AppointmentsbyDayDistributionsValues', ) 
                        ]
                        data.update({'Appointments': distributions})
                    elif appointments_dist is not None:
                        dist_params = {'name': str(name) + '_Appointments',
                                       'mean': data.get('AppointmentsDistributionValue'),
                                       'support': list(data.get('AppointmentsDistributionValues', {}).keys()),
                                       'probabilities': list(data.get('AppointmentsDistributionValues', {}).values()), }
                        distribution = create_distribution_object(appointments_dist, dist_params)
                        data.update({'Appointments': distribution})
                    patient_grp_ab_data = self.ab_data.copy().loc[[grp],:] if grp in self.ab_data.index else self.ab_data.copy().loc[['default'], :]
                elif data['Type'] == 'Bulk':
                    patient_grp_data = data.get('BulkUnits')
                    bulk_units_dist = data.get('BulkUnitsDistributions')
                    if bulk_units_dist is not None:
                        patient_grp_data = data.get('BulkUnitsDistributionsValues')
                        coefficients_by_day = data.get('BulkUnitsDistributionsCoefficientsbyDayValues', [1.])
                        distributions = [
                            {bg: create_distribution_object(bulk_units_dist, {'name': str(name) + '_BulkUnits',
                                                                           'mean': val_or_dict * day_coefficient if type(val_or_dict) is not dict else 0,
                                                                           'support': list(val_or_dict.keys()) if type(val_or_dict) is dict else [int(val_or_dict)],
                                                                           'probabilities': list(val_or_dict.values()) if type(val_or_dict) is dict else [1.],
                                                                       }) for bg, val_or_dict in data.get('BulkUnitsDistributionsValues',).items()
                             } for day_coefficient in coefficients_by_day
                        ]
                        data.update({'BulkUnits': distributions})                    
                    patient_grp_ab_data = self.ab_data.copy().loc[[grp],:] if grp in self.ab_data.index else self.ab_data.copy().loc[['default'], :]
                    patient_grp_ab_data.loc[:, :] = 0
                patient_data.update({grp: {'phenotypes': patient_grp_data, 'alloantibodies': patient_grp_ab_data}})
            self.patient_data.update({id: patient_data})
            
    def load_manufacturing_pathways(self, filename: str = 'data/locations/supply_pathways.yml') -> dict:
        """Loads manufacturing and routing pathways data

        :param str filename: Filepath to YAML file, defaults to 'data/locations/supply_pathways.yml'
        :return: Manufacturing pathways data as a dictionary
        """
        with open(filename) as f:
            a: dict[str, dict] = yaml.load(f, Loader=yaml.FullLoader)
        self.location_data.update({'manufacturing_routing': a})

    def load_immunogenicity(self, filename='data/immune_risks/immunogenicity.tsv') -> pd.DataFrame:
        """Loads immunogenicity data

        :param str filename: Filepath to TSV file, defaults to 'data/immune_risks/immunogenicity.tsv'
        :return: Immunogenicity data as a pandas DataFrame
        """
        df = pd.read_csv(filename, sep='\t')
        self.immunogenicities = df
        return self.immunogenicities

    def load_mismatch_parameters(self, filename='data/immune_risks/mismatch_weights.tsv') -> pd.DataFrame:
        """Loads mismatch weights data
        
        :param str filename: Filepath to TSV file, defaults to 'data/immune_risks/mismatch_weights.tsv'
        :return: Mismatch weights data as a pandas DataFrame
        """
        df = pd.read_csv(filename, sep='\t', index_col=0)
        self.mismatch_parameters = {col_name: df.loc[:, [col_name]].transpose() for col_name in df.columns}
        return self.mismatch_parameters.get('mismatch_expert_risk')

    def load_alloantibodies(self, filename: str = 'data/antibody_frequencies/bayes_alloAb_frequencies.tsv') -> pd.DataFrame:
        """Loads alloantibodies data

        :param filename: Filepath to TSV file, defaults to 'data/antibody_frequencies/bayes_alloAb_frequencies.tsv'
        :type filename: str
        :return: Alloantibodies data as a pandas DataFrame
        :type return: pd.DataFrame
        """
        df = pd.read_csv(filename, sep='\t', index_col=0,)
        self.ab_data = df
        return self.ab_data.loc[['default'], :].copy()
    
    def load_transportation_data(self, courier_costs: str, transit_times: list[str]) -> None:
        """Loads transportation data

        :param str courier_costs: Filepath to TSV file containing courier costs
        :param list transit_times: List of filepaths to TSV files containing transit time data
        """
        try:
            df = pd.read_csv(courier_costs, sep='\t', index_col=0)
        except (FileNotFoundError, ValueError):
            logger.warning(f'Could not load courier costs data from {courier_costs}.')
            df = None
        self.courier_data_df = df

        assert len(transit_times) == 7, "Transit times should be a list of 7 files, one for each day of the week."
        self.transit_time_data_dfs = []
        for transit_time in transit_times:
            try:
                df = pd.read_csv(transit_time, sep='\t', index_col=0)
            except (FileNotFoundError, ValueError):
                logger.warning(f'Could not load transit time data from {transit_time}.')
                df = None
            self.transit_time_data_dfs.append(df)

    def save_output(self, manager: SimulationManager, antigens: list,
                    sim_name: str, watched_phenotypes_names: list,
                    computation_times: bool, num_objectives: int,
                    objectives_names: list):
        """Save simulation output to disk."""
        folder = self.output_root_folder
        if self.is_tuning:
            os.makedirs(folder, exist_ok=True)
        ANTIGENS = antigens
        
        pad = [0, 0, 0]
        allo_padded = np.hstack(([pad, pad], np.vstack(manager.allo['total'])))
        mismatch_total = np.vstack(manager.mismatches['total'])
        subs_total = np.vstack(manager.subs['total'])
        scd_short_padded = np.full((2, len(ANTIGENS)), np.array(manager.scd_shorts)[:, None])
        all_short_padded = np.full((2, len(ANTIGENS)), np.array(manager.all_shorts)[:, None])
        allo_group_indices = []
        allo_group_stacked = []
        # Loop over each patient group except 'total'
        for group_name, allo_values in manager.allo.items():
            if group_name == 'total':
                continue
            allo_padded_grp = np.hstack(([pad, pad], np.vstack(allo_values)))
            mismatch_padded = np.vstack(manager.mismatches[group_name])
            subs_padded = np.vstack(manager.subs[group_name])
            index_grp = [f'mismatch_{group_name}_avg', f'mismatch_{group_name}_stderr',
                         f'allo_{group_name}_avg', f'allo_{group_name}_stderr',
                         f'subs_{group_name}_avg', f'subs_{group_name}_stderr',]
            to_stack = [mismatch_padded, allo_padded_grp, subs_padded,]
            allo_group_indices.extend(index_grp)
            allo_group_stacked.extend(to_stack)
        index = ['mismatch_avg', 'mismatch_stderr', 'allo_avg', 'allo_stderr', 'subs_avg', 'subs_stderr',
                'short_avg', 'short_stderr', 'all_short_avg', 'all_short_stderr']
        index.extend(allo_group_indices)
        stacked = np.vstack((mismatch_total, allo_padded, subs_total, scd_short_padded, all_short_padded,
                             *allo_group_stacked))
        df = pd.DataFrame(stacked, columns=ANTIGENS, index=index)

        now = datetime.datetime.now()
        file = os.path.join(folder, sim_name + now.strftime('%H-%M_output.tsv'))

        df.to_csv(file, sep='\t')
        print(f'Output written to {file}')
        
        obj_cols = ['alloimmunisations', 'scd_shortages', 'expiries', 'all_shortages', 'moves',
                    'scd_moves', 'other_moves',] 
        obj_stock_cols = ['O_neg_level', 'O_pos_level', 'O_level']
        obj_mm_cols = ['D_subs_num_patients', 'ABO_subs_num_patients', 'ABOD_subs_num_patients']
        all_of_the_cols = obj_cols + obj_stock_cols + obj_mm_cols

        if len(all_of_the_cols) != manager.objs.shape[1]:
            warnings.warn(f'Number of objective columns {len(all_of_the_cols)} does not match number of objectives {manager.objs.shape[1]}.'+
                          ' This may lead to mislabelling of objectives.',
                          RuntimeWarning)
            if len(all_of_the_cols) < manager.objs.shape[1]:
                extra_cols = [f'objective_{i}' for i in range(len(all_of_the_cols), manager.objs.shape[1])]
                obj_cols.extend(extra_cols)
            else:
                obj_cols = obj_cols[:len(obj_cols) - (len(all_of_the_cols) - manager.objs.shape[1])]
                
        if self.is_tuning:
            df_obj = pd.DataFrame(manager.objs, columns=obj_cols+obj_stock_cols+obj_mm_cols)
            file5 = os.path.join(folder, sim_name + now.strftime('%d_%H-%M_objectives.tsv'))
            df_obj.to_csv(file5, sep='\t', index=False)
            objectives_to_use = obj_cols if objectives_names is None else objectives_names
            objectives_to_use = objectives_to_use[:num_objectives]
            objectives_values = df_obj[objectives_to_use].mean(axis=0).values
            return objectives_values
        else:
            stocks = np.hstack(manager.stocks)
            stock_cols = watched_phenotypes_names + ['total']
            # stock_cols = kwargs.get('watched_phenotypes_names', ['O-']) + ['total']
            full_stock_cols = stock_cols + ['_se_' + a for a in stock_cols]
            df2 = pd.DataFrame(stocks, columns=full_stock_cols)

            file2 = os.path.join(folder, sim_name + now.strftime('%H-%M_stocks.tsv'))

            df2.to_csv(file2, sep='\t')
            
            # location_ids = manager.simulations[0].locations.supply_locations_ids
            location_ids = manager.supply_location_ids
            location_names = [self._supply_names[loc_id] for loc_id in location_ids]
            
            stocks_locs_array_dict = dict(stock_loc_means=manager.stocks_loc[0],
                                          stock_loc_stderrs=manager.stocks_loc[1],
                                          avail_stock_loc_means=manager.avail_stocks[0],
                                          avail_stock_loc_stderrs=manager.avail_stocks[1],
                                          watched_phenotypes_names=watched_phenotypes_names,
                                          locations=location_names)
            file21 = os.path.join(folder, sim_name + now.strftime('%H-%M_stocks_locs.npz'))
            np.savez_compressed(file21, **stocks_locs_array_dict)

            cols = [' to '.join(com) for com in list_of_permutations(
                [('O-', 'O+', 'B-', 'B+', 'A-', 'A+', 'AB-', 'AB+')] * 2)]
            df3 = pd.DataFrame(manager.abo_cm, columns=cols)
            file3 = os.path.join(folder, sim_name + now.strftime('%H-%M_abocm.tsv'))
            df3.to_csv(file3, sep='\t')
            
            df3_1 = pd.DataFrame(manager.abod_mm, columns=cols)
            file3_1 = os.path.join(folder, sim_name + now.strftime('%H-%M_abodmm_subs.tsv'))
            df3_1.to_csv(file3_1, sep='\t')
            
            df3_2_cols = ['D_substitutions', 'ABO_substitutions', 'ABOD_substitutions']
            df3_2 = pd.DataFrame(manager.pats_subs, columns=df3_2_cols)
            file3_2 = os.path.join(folder, sim_name + now.strftime('%H-%M_abodmm_pats_subs.tsv'))
            df3_2.to_csv(file3_2, sep='\t')

            file4 = os.path.join(folder, sim_name + now.strftime('%H-%M_failures.json'))

            with open(file4, 'w+') as f4:
                json.dump(manager.failures, f4, indent=2)
                
            objectives = manager.objs
            df_obj = pd.DataFrame(objectives, columns=obj_cols+obj_stock_cols+obj_mm_cols)
            file5 = os.path.join(folder, sim_name + now.strftime('%H-%M_objectives.tsv'))
            df_obj.to_csv(file5, sep='\t', index=False)

            file6 = os.path.join(folder, sim_name + now.strftime('%H-%M_age_distributions.npz'))
            age_distributions = dict(total_age_dist=manager.ages[0], total_age_dist_stderr=manager.ages[1],
                                     location_age_dists=manager.locs_ages[0], location_age_dists_stderr=manager.locs_ages[1],
                                     locations=location_names,
                                     watched_phenotypes_names=watched_phenotypes_names,
                                     location_pheno_age_dists=manager.loc_phen_ages[0],
                                     location_pheno_age_dists_stderr=manager.loc_phen_ages[1],)
            array_names = watched_phenotypes_names
            for i, name in enumerate(array_names):
                age_distributions.update({name: manager.phen_ages[0][i], name + '_stderr': manager.phen_ages[1][i]})
            age_distributions.update(
                {'age_dist_given_to_scd': manager.scd_ages[0],
                'age_dist_given_to_scd_stderr': manager.scd_ages[1]})
            np.savez_compressed(file6, **age_distributions)
            
            record_computation_times = computation_times
            # record_computation_times = kwargs.get('computation_times', False)
            if record_computation_times:
                file7 = os.path.join(folder, sim_name + now.strftime('%H-%M_computation_times.tsv'))
                np.savetxt(file7, manager.computation_times, delimiter='\t')
            # Moves    
            file9 = os.path.join(folder, sim_name + now.strftime('%H-%M_moves.npz'))
            np.savez_compressed(file9, **manager.moves)
            # Immediately unmet requests
            file10 = os.path.join(folder, sim_name + now.strftime('%H-%M_unmet_requests.npz'))
            np.savez_compressed(file10, **manager.unmet_requests)
            # Units that expired
            file11 = os.path.join(folder, sim_name + now.strftime('%H-%M_expiry_records.npz'))
            np.savez_compressed(file11, **manager.expiry_records)
            # Record of all the matches
            file12 = os.path.join(folder, sim_name + now.strftime('%H-%M_matches.npz'))
            np.savez_compressed(file12, **manager.matches)
            # Steady state xA
            file13 = os.path.join(folder, sim_name + now.strftime('%H-%M_steady_state_xA.npz'))
            np.savez_compressed(file13, mean=manager.steady_state_xA[0], stderr=manager.steady_state_xA[1])

    
def population_phenotype(config_file, black_pop_ratio) -> pd.DataFrame:
    """Create a population phenotype table.
    
    This function reads the blood group distribution from a JSON file
    and returns a pandas DataFrame with the population phenotype table.
    Each phenotype is in decimal form and has a frequency associated with it
    which is calculated based on the ratio of the White and Black populations.
    :param str config_file: Path to the JSON file containing keys for blood group systems
    and values for filenames, antigens, and which antigens to include.
    :param float black_pop_ratio: The fraction of population that is Black.
    :return DataFrame: A pandas DataFrame containing the population phenotype table.
    """
    pop_ratio = np.array([[1 - black_pop_ratio], [black_pop_ratio]])
    with open(config_file) as json_file:
        antigen_info = yaml.load(json_file, Loader=yaml.FullLoader)

    phen, freq_pop, antigens = bloodgroup_frequency(antigen_info, pop_ratio)

    pows = np.arange(len(antigens) - 1, -1, -1)
    ints = 2 ** pows
    antints = phen.dot(ints[:, None])
    df = pd.DataFrame(antints, columns=['phenotype_decimal']).astype(int)
    df['frequencies'] = freq_pop
    return df
    
    
def dummy_population_phenotypes(config_file):
    """Create a dummy population phenotype table.
    
    :param config_file: Path to the TSV file containing the major blood group distribution in the population.
    :return: A pandas dataframe containing the population phenotype table.
    """
    df = pd.read_csv(config_file, sep='\t')
    df = df.drop(columns=['ABOD'])
    return df


def create_distribution_object(dist_type: str, params: dict,) -> rv_discrete_frozen:
    """Create a scipy distribution object.
    
    :param str dist_type: The type of distribution to create.
    :param dict params: The parameters of the distribution.
    :return: A scipy distribution object.
    """
    if dist_type.replace(' ', '').lower() == 'poisson':
        frozen_dist = stats.poisson(params.get('mean'))
    elif dist_type.replace(' ', '').lower() == 'custom':
        probabilities = np.array(params['probabilities'])
        probabilities = probabilities / np.sum(probabilities)
        frozen_dist = stats.rv_discrete(name=params['name'],
                                        values=(params['support'], probabilities),)
    elif dist_type.replace(' ', '').lower() == 'discreteuniform':
        frozen_dist = stats.randint(min(params['support']), max(params['support']) + 1)
    elif dist_type.replace(' ', '').lower() == 'continuousuniform':
        frozen_dist = stats.uniform(loc=min(params['support']), scale=max(params['support']) - min(params['support']))
    else:
        raise ValueError(f'Unknown distribution type: {dist_type}')
    return frozen_dist
    
