import argparse
import copy
import os
import glob
import json
import subprocess
import time
from PisaFlexibleClient import initialise_env
import signal
import psutil


class Checker(object):
    def __init__(self, working_dir, isa_path, theory_file, port=9000):
        self.working_dir = working_dir
        self.isa_path = isa_path
        self.theory_file = theory_file
        self.port = port
        self.env = None
        # Initialize environment
        env = self._initialize()
        if env.successful_starting:
            print('finally successfully initialize the environment')
            self.success_env = True
            self.env = env
        else:
            print('failure initialize the environment')
            self.success_env = False

    def _initialize(self):
        print("Initializing environment")
        print("ISA_PATH: %s" % self.isa_path)
        print("THEORY_FILE: %s" % self.theory_file)
        print("WORKING_DIR: %s" % self.working_dir)
        env = initialise_env(
            self.port,
            working_directory=self.working_dir,
            isa_path=self.isa_path,
            theory_file_path=self.theory_file
        )
        if env.successful_starting:
            print("Start doing post env initialising environment")
            env.post('<initialise>')
        else:
            print('initialize_env function failed')
        return env

    def _exit(self, env):
        try:
            env.post('exit')
        except:
            print("env.post('exit') timed out")
            pass
        os.system("ps aux | grep Isabelle | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
        os.system("ps aux | grep poly | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")

    def _parse_output(self, obs):
        """Parse the sledgehammer output, otherwise return an empty string"""
        if '<hammer>' in obs:
            output = obs.split('<hammer>')[0]
        else:
            output = ''
        return output

    def _run_step(self, step, i, tls_name, env):
        try:
            obs, reward, done, metadata = env.step_to_top_level_state(
                action=step,
                tls_name=tls_name,
                new_name='default_%d' % i
            )
        except:
            pass
        error = None
        if 'error:' in obs or 'Step error' in obs or 'Unknown error' in obs:
            error = obs
        return obs, reward, done, metadata, error

    def _run_sledgehammer(self, step, i, tls_name, env):
        # First try heuristics
        for heuristic in ['by auto', 'by simp', 'by blast', 'by fastforce', 'by force', 'by eval', 'by presburger', 'by sos', 'by arith', 'by linarith', 'by (auto simp: field_simps)']:
            step_ = step.replace('sledgehammer', heuristic)
            obs, reward, done, metadata, error = self._run_step(step_, i, tls_name, env)
            if error is None:
                obs = '%s <hammer> %s' % (heuristic, obs)
                return obs, reward, done, metadata, error
        # Try sledgehammer
        return self._run_step(step, i, tls_name, env)

    def check_common_logic(self, steps):
        env = self.env
        done = False
        reason = ''
        success = False
        step_results = []
        tls_name = 'default'
        for i, step in enumerate(steps):
            try:
                if 'sledgehammer' in step:
                    obs, reward, done, metadata, error = self._run_sledgehammer(step, i, tls_name, env)
                else:
                    obs, reward, done, metadata, error = self._run_step(step, i, tls_name, env)
                step_results.append(dict(index=i, step=step, output=self._parse_output(obs)))
                if error is not None:
                    reason = error
                    success = False
                    done = False
                    break
                else:
                    if 'sledgehammer' in step:
                        print('sledgehammer output: {0}'.format(obs))
            except:
                # Timeout - end the proof attempt
                success = False
                done = False
                reason = 'timeout (%d)' % len(step_results)
                step_results.append(dict(index=i, step=step, output=''))
                break

            # Change when successful
            tls_name = 'default_%d' % i

        if done and reward == 1.0:
            success = True

        result = {
            'success': success,
            'reason': reason,
            'num_steps': len(steps),
            'last_step': len(step_results),
            'step_results': step_results
        }
        return result

    def check_minif2f(self, formal, formal_statement=None, formal_proof=None):
        env = self.env
        # Wrap and parse theorem
        start_time = time.time()
        theory = Checker.minif2f_wrap_theorem(formal)
        # steps = Checker.get_parsed(env, theory)
        steps = Checker.get_parsed_hacky(formal_statement, formal_proof)
        print('before common logic takes {0} seconds'.format(time.time() - start_time))
        result = self.check_common_logic(steps)
        # Exit environment
        # self._exit(env)
        return result

    @staticmethod
    def minif2f_wrap_theorem(theorem):
        return 'theory Interactive imports HOL.HOL Complex_Main "HOL-Library.Code_Target_Numeral" "HOL-Library.Sum_of_Squares" "Symmetric_Polynomials.Vieta" "HOL-Computational_Algebra.Computational_Algebra" "HOL-Number_Theory.Number_Theory" \n begin\n\nfunction digits_in_base :: \"nat \\<Rightarrow> nat \\<Rightarrow> nat list\" where \n  \"digits_in_base n k = (if n div k = 0 \\<or> k=1\n      then [n] else (n mod k) # (digits_in_base (n div k) k))\"\n  by auto\ntermination \n  by (relation \"measure fst\") (auto simp add: div_greater_zero_iff)\n %s' % theorem

    @staticmethod
    def wrap_theorem(theorem):
        return 'theory Interactive imports Complex_Main \n "HOL-Computational_Algebra.Computational_Algebra" \n "HOL-Number_Theory.Number_Theory" \n begin\n%s' % theorem

    @staticmethod
    def get_parsed(env, theory, tls_name='default'):
        steps = env.post(f"<parse text> ${theory}")
        steps = steps.split('<SEP>')
        steps = [s for s in steps if s.strip() != '']
        print(steps)
        # remove weird '$' step and whitespace steps
        steps = [s for s in steps if s != '$' and s.strip() != '']
        return steps

    @staticmethod
    def get_parsed_hacky(formal_statement, formal_proof):
        if 'digits_in_base' in formal_statement:
            preceding_steps = ['$',
             'theory Interactive imports HOL.HOL Complex_Main "HOL-Library.Code_Target_Numeral" "HOL-Library.Sum_of_Squares" "Symmetric_Polynomials.Vieta" "HOL-Computational_Algebra.Computational_Algebra" "HOL-Number_Theory.Number_Theory" \n begin',
             'function digits_in_base :: "nat \\<Rightarrow> nat \\<Rightarrow> nat list" where \n  "digits_in_base n k = (if n div k = 0 \\<or> k=1\n      then [n] else (n mod k) # (digits_in_base (n div k) k))"',
             'by auto', 'termination', 'by (relation "measure fst") (auto simp add: div_greater_zero_iff)',]
        else:
            preceding_steps = ['$',
             'theory Interactive imports HOL.HOL Complex_Main "HOL-Library.Code_Target_Numeral" "HOL-Library.Sum_of_Squares" "Symmetric_Polynomials.Vieta" "HOL-Computational_Algebra.Computational_Algebra" "HOL-Number_Theory.Number_Theory" \n begin',]
        if formal_proof == 'using assms sledgehammer':
            steps = preceding_steps + [formal_statement, 'using assms', 'sledgehammer']
        else:
            assert formal_proof == 'sledgehammer'
            steps = preceding_steps + [formal_statement, 'sledgehammer']
        # remove weird '$' step and whitespace steps
        steps = [s for s in steps if s != '$' and s.strip() != '']
        return steps

    @staticmethod
    def get_sub_dir(thy_file_path):
        # get the isabelle sub directory to work with
        thy_file_path_split = thy_file_path.split('/')
        sub_dir = thy_file_path_split[0]
        if sub_dir == 'HOL':
            if len(thy_file_path_split) > 2:
                sub_dir_next = thy_file_path_split[1]
                sub_dir = sub_dir + '/' + sub_dir_next + '/'
        return sub_dir


def initialize_pisa_env(port):
    start_time_single = time.time()
    working_directory = '/home/jz563/afp-2021-10-22/thys/Symmetric_Polynomials/'
    theory_file_path = '/home/jz563/afp-2021-10-22/thys/Symmetric_Polynomials/Interactive.thy'
    isa_path = "/home/jz563/Isabelle2021/"
    # clean up stuff just in case
    if os.path.exists('sbt_ready_{0}.txt'.format(port)):
        os.system('rm sbt_ready_{0}.txt'.format(port))
    os.system("ps aux | grep Isabelle | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
    os.system("ps aux | grep poly | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
    os.system("ps aux | grep sbt | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
    sbt_ready = False
    environment_success = False
    failure_counter = 0
    while not environment_success and failure_counter <= 10:
        print('starting the server')
        # print('deleting sbt bg-jobs folder')
        # os.system('rm -rf target/bg-jobs/')
        sub = subprocess.Popen('sbt "runMain pisa.server.PisaOneStageServer{0}" | tee sbt_ready_{0}.txt'.format(port), shell=True)
        pid = sub.pid
        while not sbt_ready:
            if os.path.exists('sbt_ready_{0}.txt'.format(port)):
                with open('sbt_ready_{0}.txt'.format(port), 'r') as f:
                    file_content = f.read()
                if 'Server is running. Press Ctrl-C to stop.' in file_content and 'error' not in file_content:
                    print('sbt should be ready')
                    sbt_ready = True
        print(f"Server started with pid {pid}")
        time.sleep(3)
        checker = Checker(
            working_dir=working_directory,
            isa_path=isa_path,
            theory_file=theory_file_path,
            port=port,
        )
        if checker.success_env:
            print('escaping the while loop')
            environment_success = True
        else:
            print('restarting the while loop')
            failure_counter += 1
            try:
                parent = psutil.Process(pid)
                children = parent.children(recursive=True)
                for process in children:
                    process.send_signal(signal.SIGTERM)
                parent.send_signal(signal.SIGTERM)
            except psutil.NoSuchProcess:
                pass
            # delete sbt ready txt
            os.system('rm sbt_ready_{0}.txt'.format(port))
            os.system("ps aux | grep Isabelle | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
            os.system("ps aux | grep poly | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
            os.system("ps aux | grep sbt | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
            sbt_ready = False
            environment_success = False
    if not environment_success:
        print('environment still cannot be initialized')
        raise NotImplementedError
    print('initializing environment required {0} seconds'.format(time.time() - start_time_single))
    return checker, pid


def exit_pisa_env(port, pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for process in children:
            process.send_signal(signal.SIGTERM)
        parent.send_signal(signal.SIGTERM)
    except psutil.NoSuchProcess:
        pass
    # delete sbt ready txt clean up things just in case
    os.system('rm sbt_ready_{0}.txt'.format(port))
    os.system("ps aux | grep Isabelle | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
    os.system("ps aux | grep poly | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
    os.system("ps aux | grep sbt | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")


def evaluate_one_proof(proof_data, save_path):
    start_time_single = time.time()
    formal_statement = proof_data["formal_statement"]
    # sanitized the statement
    formal_statement = ' '.join(formal_statement.split())
    generated_proof = proof_data["generated_formal_proof"]
    print('statement to prove: {0}'.format(formal_statement))
    result = checker.check_minif2f(formal_statement + "\n" + generated_proof, formal_statement=formal_statement, formal_proof=generated_proof)
    if result == 'statement_not_exist':
        raise NotImplementedError
    else:
        proof_data.update({"proof_checking_success": result["success"], "proof_checking_result": result})
    print('checking success {0} for {1}'.format(proof_data['proof_checking_success'], save_path))
    with open(save_path, 'w') as f:
        json.dump(proof_data, f)
    print('checking single required {0} seconds'.format(time.time() - start_time_single))


parser = argparse.ArgumentParser(description='')
parser.add_argument('-response_path', type=str, required=True, help='')
parser.add_argument('-save_dir', type=str, default='')
parser.add_argument('-start_index', type=int, default=0)
parser.add_argument('-end_index', type=int, default=-1)
parser.add_argument('-port', type=int, default=8000)

args = parser.parse_args()
start_time_overall = time.time()
print(vars(args))

if args.save_dir == '':
    save_dir = args.response_path.rsplit('/', 1)[0].replace('responses_chunks', 'evaluation_chunks/')
else:
    save_dir = args.save_dir
with open(args.response_path, 'r') as f:
    data = f.readlines()
data = [json.loads(e) for e in data]
print('saving dir', save_dir)
print('total {0} proofs to check on this machine'.format(len(data)))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
port = args.port
checker, pid = initialize_pisa_env(port)
proofs_checked_so_far = 0
for i in range(len(data)):
    print('currently checking {0}th proof'.format(i))
    if i < args.start_index:
        continue
    if args.end_index != -1 and i >= args.end_index:
        break
    evaluate_one_proof(data[i], os.path.join(save_dir, '{0}.json'.format(i)))
    proofs_checked_so_far += 1
exit_pisa_env(port, pid)
