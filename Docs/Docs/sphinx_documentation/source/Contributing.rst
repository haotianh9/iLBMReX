.. _Chap:Contributing:

Contributing
===============

Development Model
-----------------

iLBMReX development follows these principles:

* New features are merged into the ``main`` branch using Pull Requests (PRs).

* Bug fixes, questions, and feature contributions are welcome!

  * Bugs should be reported through `GitHub Issues <https://github.com/haotianh9/lattice_boltzmann_method/issues>`_.
  * Questions can be asked through `GitHub Discussions <https://github.com/haotianh9/lattice_boltzmann_method/discussions>`_.
  * All contributions should be done via pull requests.
  * A pull request should be generated from your fork and target the ``main`` branch.

  When merging, commits are squashed to maintain a clean history. **Please ensure
  that your PR title and description are clear and descriptive**, as these will
  be used for the squashed commit message.

  By contributing code, you grant a non-exclusive, royalty-free perpetual license
  to use, modify, and distribute your contributions.

Git Workflow
------------

iLBMReX uses `git <https://git-scm.com>`_ for version control. If you are new to
git, see these resources:

- `Learn git with Bitbucket <https://www.atlassian.com/git/tutorials/learn-git-with-bitbucket-cloud>`_
- `git - the simple guide <http://rogerdudler.github.io/git-guide/>`_

The basic workflow is:

1. Fork the main repository
2. Implement your changes on a new branch ``<branch_name>``
3. Push your commits to your fork
4. Create a Pull Request from your branch to ``main`` on the main iLBMReX repository

.. _sec:forking:

Making Your Own Fork
^^^^^^^^^^^^^^^^^^^^^

To make a fork of the main repository, press the fork button on the
`iLBMReX GitHub page <https://github.com/haotianh9/lattice_boltzmann_method>`_.

Then clone your fork locally. We recommend using SSH authentication to avoid
entering your GitHub password repeatedly:

.. code:: shell

  git clone git@github.com:<myGithubUsername>/lattice_boltzmann_method.git

  # Navigate into your repo and add a remote for the main repository
  cd lattice_boltzmann_method
  git remote add upstream https://github.com/haotianh9/lattice_boltzmann_method
  git fetch upstream

For SSH setup instructions, see the
`GitHub SSH documentation <https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh>`_.

Alternatively, use HTTPS authentication:

.. code:: shell

  git clone https://github.com/<myGithubUsername>/lattice_boltzmann_method.git

  # Navigate into your repo and add a remote for the main repository
  cd lattice_boltzmann_method
  git remote add upstream https://github.com/haotianh9/lattice_boltzmann_method
  git fetch upstream

To sync your local ``main`` branch with the upstream repository:

.. code:: shell

  git checkout main
  git pull upstream main

Making Code Changes
^^^^^^^^^^^^^^^^^^^^

Do **not** make changes directly in the ``main`` branch. Instead, create a new
branch based on ``main`` to isolate your changes:

.. code:: shell

  git checkout main
  git checkout -b <branch_name>

Choose a descriptive branch name that reflects your changes (e.g., ``fix_ib_interpolation``,
``add_sphere_geometry``).

Stage and commit your changes:

.. code:: shell

  git add <file_I_created> <and_file_I_modified>
  git commit -m "Clear 50-character description of the changes"

Write descriptive commit messages to help identify bugs and track development history.

Push your commits to your fork:

.. code:: shell

  git push -u origin <branch_name>

To synchronize your branch with the upstream ``main`` branch (useful when the
main repository has been updated):

.. code:: shell

  git fetch upstream
  git merge upstream/main

Fix any conflicts that arise. Do **not** merge your branch back into your local
``main`` branch, as this will cause your ``main`` to diverge from the upstream
version. Instead, use a Pull Request.

Submitting a Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^^

After pushing your changes to your fork, GitHub will show a banner suggesting
you create a pull request. Click ``compare & pull request``.

Provide a clear title and description for your PR:

- **What feature/fix are you proposing and why?**
- **How did you implement it?** (created a new class, modified existing function, etc.)
- **Any relevant details?** (performance tests, new figures, benchmarks, etc.)

Click ``Create pull request``. Your PR will now be visible and reviewable on
the main repository.

**Guidelines for Pull Requests:**

* Keep PRs small and focused. Large PRs are difficult and time-consuming to review.
* Submit fixes and features as separate PRs when possible.
  - Typo fixes in one PR
  - Bug fixes in another PR
  - Feature implementations as focused PRs
* Use ``draft`` PRs for work-in-progress that you want to discuss or get feedback on.

Once reviewed and approved, your PR will be merged into ``main``. You can then
delete your branch:

.. code:: shell

  git branch -D <branch_name>  # Delete local copy
  git push origin --delete <branch_name>  # Delete remote copy on your fork

iLBMReX Coding Style Guide
---------------------------

Code Guidelines
^^^^^^^^^^^^^^^^^

iLBMReX developers should adhere to these guidelines:

* Use **4 spaces** for indentation, not tabs.
* Use curly braces for single-statement blocks:

  .. code::

     for (int n=0; n<10; ++n) {
         amrex::Print() << "Like this!";
     }

  or on one line:

  .. code::

     for (int n=0; n<10; ++n) { amrex::Print() << "Like this!"; }

  but **not**:

  .. code::

     for (int n=0; n<10; ++n) amrex::Print() << "Not like this.";

* Add a space after the function name and before the opening parenthesis
  in function declarations (but not in function calls):

  .. code::

     void CorrectFunctionName (int input);  // Declaration

     CorrectFunctionName(value);  // Call

  This makes it easy to locate function definitions with ``grep``.

* Member variables should be prefixed with ``m_``:

  .. code::

     Real m_velocity;
     int m_refinement_level;

These guidelines should be followed in new contributions. However, avoid making
stylistic changes to unrelated code in your PRs.

API Documentation Using Doxygen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

API documentation is generated from Doxygen comments in the source code.
Doxygen comment blocks should precede the namespace, class, function, or
variable being documented:

.. code::

  /**
   * \brief Brief description of the function.
   *
   * \param[in] variable Description of input variable.
   * \param[inout] data Description of data (read and modified).
   *
   * Longer description can be included here.
   */
  void MyFunction (int variable, amrex::MultiFab& data);

See the `Doxygen Manual <https://www.doxygen.nl/manual/>`_ for detailed
formatting information.
